from flask import Blueprint, render_template, Response, redirect, url_for, request
import cv2

video_bp = Blueprint('video', __name__)

is_recording = False
video_writer = None

questions_dict = {
    "programmer": [
        "Что такое ООП?",
        "Какие структуры данных вы знаете?",
        "Как работает HTTP?",
        "Что такое SQL и NoSQL базы данных?",
        "Какие у вас любимые языки программирования?"
    ],
    "other": [
        "Расскажите о себе.",
        "Почему вы хотите работать в нашей компании?",
        "Как вы справляетесь со стрессом?",
        "Какие у вас сильные и слабые стороны?",
        "Как вы видите себя через 5 лет?"
    ]
}

def generate_frames():
    global is_recording, video_writer

    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        if is_recording and video_writer is not None:
            video_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    if video_writer is not None:
        video_writer.release()

@video_bp.route('/video_page')
def video_page():
    return render_template('video.html')

@video_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@video_bp.route('/start_recording')
def start_recording():
    global is_recording, video_writer

    if not is_recording:
        is_recording = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    return redirect(url_for('video.video_page'))

@video_bp.route('/stop_recording')
def stop_recording():
    global is_recording, video_writer

    if is_recording:
        is_recording = False
        video_writer.release()
        video_writer = None

    return redirect(url_for('video.video_page'))

@video_bp.route('/start_interview', methods=['POST'])
def start_interview():
    category = request.form.get('category', 'other')
    questions = questions_dict.get(category, [])

    return render_template('video.html', questions=questions)