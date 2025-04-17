from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
from hashlib import sha256
import base64
import json
from datetime import datetime
from io import BytesIO
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Layer
from tflocalpattern.layers import LDP

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://wei:tanwei1103@localhost/fyp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/path/to/your/upload/directory' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg',"jfif"}

db = SQLAlchemy(app)

# Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    face_data = db.Column(db.Text)
    is_approved = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    vote_area_id = db.Column(db.Integer, db.ForeignKey('vote_areas.id'))

class Candidate(db.Model):
    __tablename__ = 'candidates'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    information = db.Column(db.Text)
    vote_area_id = db.Column(db.Integer, db.ForeignKey('vote_areas.id'))

class Vote(db.Model):
    __tablename__ = 'votes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidates.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    blockchain_reference = db.Column(db.Text)

class VoteArea(db.Model):
    __tablename__ = 'vote_areas'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)

def setup_database(app):
    with app.app_context():
        db.create_all()

setup_database(app)


class CustomLDP(LDP):
    def __init__(self, mode='single', alpha='0', **kwargs):
        super(CustomLDP, self).__init__(mode=mode, alpha=alpha, **kwargs)
        self.mode = mode
        self.alpha = alpha

    def build(self, input_shape):
        super(CustomLDP, self).build(input_shape)  

    def call(self, inputs):
        return super(CustomLDP, self).call(inputs)

    def get_config(self):
        config = super(CustomLDP, self).get_config()
        config.update({'mode': self.mode, 'alpha': self.alpha})
        return config
    

model = tf.keras.models.load_model('checkpoint1.h5', custom_objects={'CustomLDP': CustomLDP})
NCLASS = 50

def allowed_file(filename):
    """Check if the filename's extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('You need to be logged in to view this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/home_index')
def home_index():
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash('You have been logged out.', 'success')
    # Redirect to the login page
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        vote_area_id = request.form.get('vote_area_id')
        file = request.files.get('face_image')

        if file and allowed_file(file.filename):
            # Convert the image to a base64 string
            img_buffer = BytesIO()
            file.save(img_buffer)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

            # Now you can save this base64 string to your database
            existing_user = User.query.filter_by(username=username).first()
            if not existing_user:
                new_user = User(
                    username=username, 
                    password=password, 
                    face_data=img_base64, 
                    vote_area_id=vote_area_id
                )
                db.session.add(new_user)
                db.session.commit()
                flash('Registration successful! You can now log in.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Username already exists', 'danger')
        else:
            flash('Invalid file format. Please upload an image.', 'danger')
            return redirect(url_for('register'))

    vote_areas = VoteArea.query.all()
    return render_template('register.html', vote_areas=vote_areas)


def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('You need to be logged in to view this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('login'))

def preprocess_image(image_data):
    """Preprocesses the image for the model."""
    image = Image.open(BytesIO(image_data)).convert('L')  # Convert to grayscale
    image = image.resize((94,125))  # Resize to match model's input shape
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return np.expand_dims(image_array, axis=0)

@app.route('/login', methods=['GET', 'POST'])
def login():
    face_recognition_failed = False
    username = None

    if request.method == 'POST':
        username = request.form.get('username')
        image_data = request.form.get('image')
        password = request.form.get('password')  # It's important to handle the case when this might not be provided.

        user = User.query.filter_by(username=username).first()
        if user:
            if image_data:
                # Process image data for face recognition
                try:
                    image_data = base64.b64decode(image_data.split(',')[1])
                    image_array = preprocess_image(image_data)  
                    predictions = model.predict(image_array)
                    predicted_class = np.argmax(predictions, axis=1)[0]

                    stored_face_data = preprocess_image(base64.b64decode(user.face_data))
                    similarity_score = cosine_similarity(predictions, model.predict(stored_face_data))[0][0]


                    # Debug logging (remove in production)
                    app.logger.debug(f"Similarity Score: {similarity_score}") 
                    app.logger.debug(f"Prediction : {predictions}")
                    

                    if similarity_score >= 0.8: 
                        session['logged_in'] = True
                        session['user_id'] = user.id
                        session['username'] = user.username
                        return redirect(url_for('home'))
                    else:
                        face_recognition_failed = True  # Indicate that face recognition has failed
                except Exception as e:
                    flash(f"Error processing image: {str(e)}", 'danger')
                    face_recognition_failed = True

            # If face recognition wasn't used or failed, check the password
            if not image_data or face_recognition_failed:
                if password == user.password: 
                    session['logged_in'] = True
                    session['user_id'] = user.id
                    session['username'] = user.username
                    return redirect(url_for('home'))
                else:
                    # Set face_recognition_failed to True to prompt for a password on the template
                    face_recognition_failed = True
                    flash('Face recognition failed, please use your password to login.', 'danger')
        else:
            flash('Invalid username!', 'danger')

    # The template needs to know whether to show the password field or not
    return render_template('login.html', password_required=face_recognition_failed, username=username)


@app.route('/home')
@login_required
def home():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    return render_template('home.html', user=user)

@app.route('/admin/approve', methods=['GET', 'POST'])
@login_required
def admin_approve():
    current_user = User.query.get(session['user_id'])
    if not current_user.is_admin:
        flash("You are not authorized to view this page.", "danger")
        return redirect(url_for('home'))
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        action = request.form.get('action')
        user_to_update = User.query.get(user_id)
        if user_to_update:
            if action == 'approve':
                user_to_update.is_approved = True
            elif action == 'reject':
                db.session.delete(user_to_update)
            db.session.commit()
            flash(f"User {user_to_update.username} action completed.", "success")
    unapproved_users = User.query.filter_by(is_approved=False).all()
    return render_template('admin_approve.html', unapproved_users=unapproved_users)

@app.route('/admin/add_candidate', methods=['GET', 'POST'])
@login_required
def admin_candidate():
    current_user = User.query.get(session['user_id'])
    if not current_user or not current_user.is_admin:
        flash("You are not authorized to view this page.", "danger")
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add':
            name = request.form.get('name')
            information = request.form.get('information')
            vote_area_id = request.form.get('vote_area_id', type=int)

            vote_area = VoteArea.query.get(vote_area_id)
            if vote_area:
                new_candidate = Candidate(name=name, information=information, vote_area_id=vote_area_id)
                db.session.add(new_candidate)
                db.session.commit()
                flash('New candidate added successfully.', 'success')
            else:
                flash('Vote area not found.', 'danger')
        elif action == 'remove':
            candidate_id = request.form.get('candidate_id', type=int)
            candidate_to_remove = Candidate.query.get(candidate_id)
            if candidate_to_remove:
                db.session.delete(candidate_to_remove)
                db.session.commit()
                flash('Candidate removed successfully.', 'success')
            else:
                flash('Candidate not found.', 'danger')

    candidates = Candidate.query.all()
    vote_areas = VoteArea.query.all()
    return render_template('admin_candidate.html', candidates=candidates, vote_areas=vote_areas)



@app.route('/admin/manage_vote_areas', methods=['GET', 'POST'])
@login_required
def manage_vote_areas():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add':
            name = request.form['name']
            description = request.form['description']
            new_vote_area = VoteArea(name=name, description=description)
            db.session.add(new_vote_area)
            flash('Vote area added successfully!', 'success')
        elif action == 'delete':
            vote_area_id = request.form.get('vote_area_id')
            vote_area = VoteArea.query.get(vote_area_id)
            if vote_area:
                db.session.delete(vote_area)
                flash('Vote area removed successfully!', 'success')
            else:
                flash('Vote area not found.', 'danger')
        db.session.commit()
        # Redirect to the same route to refresh the page
        return redirect(url_for('manage_vote_areas'))

    vote_areas = VoteArea.query.all()
    return render_template('manage.html', vote_areas=vote_areas)

@app.route('/admin/view_all_votes')
@login_required
def admin_view_votes():
    votes = Vote.query.order_by(Vote.timestamp.desc()).all()  # Fetch votes and order by timestamp
    return render_template('admin_view.html', votes=votes)

@app.route('/view_candidates')
@login_required
def view_candidates():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    if user:
        candidates = Candidate.query.filter_by(vote_area_id=user.vote_area_id).all()
    else:
        flash('You need to be logged in to view this page.', 'danger')
        return redirect(url_for('login'))
    
    return render_template('user_candidate.html', candidates=candidates)


@app.route('/cast_vote', methods=['GET', 'POST'])
@login_required
def cast_vote():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if not user:
        flash('User not found. Please log in again.', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        candidate_id = request.form.get('candidate_id', type=int)
        candidate = Candidate.query.get(candidate_id)
        
        if candidate and candidate.vote_area_id == user.vote_area_id:
            # Check if the user has already voted
            existing_vote = Vote.query.filter_by(user_id=user_id).first()
            
            if existing_vote:
                flash('You have already cast your vote.', 'warning')
            else:
                new_vote = Vote(user_id=user_id, candidate_id=candidate_id)
                db.session.add(new_vote)
                db.session.commit()
                flash('Your vote has been successfully cast!', 'success')
        else:
            flash('Invalid candidate or not in your voting area.', 'danger')

    # Get the list of candidates for the current user's area, regardless of GET or POST request
    candidates = Candidate.query.filter_by(vote_area_id=user.vote_area_id).all()
    
    # Render the voting form with the candidates
    return render_template('user_vote.html', candidates=candidates)


def hash_block(block):
    block_encoded = json.dumps(block, sort_keys=True).encode()
    return sha256(block_encoded).hexdigest()


@app.route('/view_my_ballot')
@login_required
def view_ballot_overview():
    vote_areas = VoteArea.query.all()
    ballot_overview = {}

    for area in vote_areas:
        votes_in_area = Vote.query.join(Candidate).filter(Candidate.vote_area_id == area.id).all()
        total_votes = len(votes_in_area)
        
        # Fetch users in the area only once for efficiency
        users_in_area_count = User.query.filter_by(vote_area_id=area.id).count()

        candidate_votes = {}
        for vote in votes_in_area:
            candidate_name = Candidate.query.get(vote.candidate_id).name
            candidate_votes[candidate_name] = candidate_votes.get(candidate_name, 0) + 1

        # Calculate vote rate only if there are users in the area
        if users_in_area_count > 0:
            vote_rate = (total_votes / users_in_area_count) * 100
        else:
            vote_rate = 0  # Or any other value you want to display when there are no users

        ballot_overview[area.name] = {'vote_rate': vote_rate, 'candidate_votes': candidate_votes}

    return render_template('user_view.html', ballot_overview=ballot_overview)


    
if __name__ == '__main__':
    app.run(debug=True)
