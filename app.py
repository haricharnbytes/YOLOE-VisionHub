import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from vision_engine import VisionEngine

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = "super_secret_yoloe_key" # Required for flashing error messages

# Configure the upload directory relative to the static folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize our YOLOE-26 Vision Engine
print("Initializing Vision Engine...")
engine = VisionEngine()

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload interface."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handles the image upload and zero-shot detection process."""
    # 1. Check if the post request has the file part and prompt
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(request.url)
    
    file = request.files['file']
    prompts = request.form.get('prompts', '').strip()

    # 2. Validate inputs
    if file.filename == '':
        flash('No image selected for uploading.')
        return redirect(url_for('index'))
    
    if not prompts:
        flash('Please enter at least one object to detect.')
        return redirect(url_for('index'))

    # 3. Process and save the file
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save original image
        file.save(input_filepath)
        
        # Define output filename
        output_filename = f"detected_{filename}"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # 4. Run the YOLOE-26 zero-shot detection
        success = engine.process_image(input_filepath, prompts, output_filepath)
        
        if success:
            # Render the template again, passing the output image path and prompts used
            return render_template(
                'index.html', 
                input_image=filename,
                output_image=output_filename,
                prompts=prompts
            )
        else:
            flash('An error occurred during image processing.')
            return redirect(url_for('index'))
            
    else:
        flash('Allowed image types are -> png, jpg, jpeg, webp')
        return redirect(url_for('index'))

if __name__ == "__main__":
    # Run the app on all available IPs so Docker can map it correctly
    app.run(host="0.0.0.0", port=5000, debug=True)