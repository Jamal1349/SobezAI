from flask import Flask, render_template, request, redirect, url_for
import os
import matplotlib.pyplot as plt
from video_processing.nitec_model import nitec_model
from expression_ssd_detect import emtion_pred
from collections import Counter
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['GRAPH_FOLDER'] = './static/graphs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "Видео не было выбрано. Пожалуйста, выберите видео.", 400
    file = request.files['video']
    if file.filename == '':
        return "Имя файла отсутствует. Пожалуйста, выберите файл.", 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        nitec_results = nitec_model(file_path, None)  # Нужен только результат
        emotion_results = emtion_pred(file_path, None)  # Нужен только результат
        nitec_results = [int(x.results[0] > 0.5) for x in nitec_results]
        nitec_graph_path = os.path.join(app.config['GRAPH_FOLDER'], 'nitec_graph.png')
        emotion_graph_path = os.path.join(app.config['GRAPH_FOLDER'], 'emotion_graph.png')
        build_nitec_graph(nitec_results, nitec_graph_path)
        build_emotion_graph(emotion_results, emotion_graph_path)
        return redirect(url_for('results'))


@app.route('/results')
def results():
    nitec_graph = os.path.join('static/graphs', 'nitec_graph.png')
    emotion_graph = os.path.join('static/graphs', 'emotion_graph.png')
    return render_template('results.html', nitec_graph=nitec_graph, emotion_graph=emotion_graph)


def build_nitec_graph(data, save_path):
    contact = sum(data)
    no_contact = len(data) - contact
    labels = ['Зрительный\nконтакт', 'Нет контакта']  # Разделяем текст на две строки
    sizes = [contact, no_contact]
    colors = ['#66c2a5', '#fc8d62']

    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops={'fontsize': 12},  # Размер текста
        labeldistance=1.1  # Расстояние меток от центра
    )
    plt.title('Процент зрительного контакта', fontsize=16)
    plt.savefig(save_path)
    plt.close()



def build_emotion_graph(data, save_path):
    data = Counter(data)
    emotions = list(data.keys())
    counts = list(data.values())
    colors = plt.cm.Paired(range(len(emotions)))
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Распределение эмоций')
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    app.run(debug=True)
