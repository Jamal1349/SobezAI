import cv2
import multiprocessing as mp
import mediapipe as mpipe
"""
from best_exp import emotion_pred
from best_ges import gesture_score"""
from best_nitec import nitec_model

mp_hands = mpipe.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def process_frame(frame):
    #result_emotion = emotion_pred(frame)
    
    result_nitec = nitec_model(frame)
    
    result_gesture = None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    #if results.multi_hand_landmarks:
        #result_gesture = gesture_score(frame)
    
    return result_nitec #result_emotion, result_gesture,"""


def video_worker(video_path, start_frame, end_frame, output_queue, n):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}.")
        output_queue.put(([], [], []))
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results_emotion = []
    results_gesture = []
    results_nitec = []
    frame_number = start_frame

    while frame_number < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % n == 0:
            try:
                result_emotion, result_gesture, result_nitec = process_frame(frame)
                results_emotion.append(result_emotion)
                results_nitec.append(result_nitec)
                if result_gesture is not None:
                    results_gesture.append(result_gesture)
            except Exception as e:
                print(f"Ошибка при обработке кадра {frame_number}: {e}")

        frame_number += 1

    cap.release()
    output_queue.put((results_emotion, results_gesture, results_nitec))


def split_video_into_chunks(video_path, num_processes):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    chunk_size = total_frames // num_processes
    chunks = []
    for i in range(num_processes):
        start_frame = i * chunk_size
        end_frame = (i + 1) * chunk_size if i < num_processes - 1 else total_frames
        chunks.append((start_frame, end_frame))

    return chunks


def video_proc(input_video_path, num_processes, n):
    available_cores = mp.cpu_count()
    print(f"Доступно ядер CPU: {available_cores}")

    if num_processes is None:
        num_processes = available_cores
    elif num_processes > available_cores:
        print(f"Предупреждение: Указано {num_processes} процессов, но доступно только {available_cores} ядер. Используется {available_cores} процессов.")
        num_processes = available_cores
    chunks = split_video_into_chunks(input_video_path, num_processes)

    output_queue = mp.Queue()

    processes = []
    for i, (start_frame, end_frame) in enumerate(chunks):
        p = mp.Process(target=video_worker, args=(input_video_path, start_frame, end_frame, output_queue, n))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results_emotion = []
    results_gesture = []
    results_nitec = []
    while not output_queue.empty():
        emotion_chunk, gesture_chunk, nitec_chunk = output_queue.get()
        results_emotion.extend(emotion_chunk)
        results_gesture.extend(gesture_chunk)
        results_nitec.extend(nitec_chunk)

    output_queue.close()
    output_queue.join_thread()
    return results_emotion, results_gesture, results_nitec
