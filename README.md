import cv2
import face_recognition

# Video kaynağı olarak kamerayı kullanmak için
video_capture = cv2.VideoCapture(0)

# Tanıtacağımız yüzün resmini yüklüyoruz ve yüz kodunu çıkarıyoruz
known_image = face_recognition.load_image_file("taninacak_yuz.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Bilinen yüzleri ve isimleri listede saklıyoruz
known_face_encodings = [known_face_encoding]
known_face_names = ["akıb mustafa özel"]

while True:
    # Video akışından bir kare alıyoruz
    ret, frame = video_capture.read()

    # Her kareyi BGR (OpenCV) formatından RGB (face_recognition) formatına dönüştürüyoruz
    rgb_frame = frame[:, :, ::-1]

    # Görüntüdeki tüm yüzleri ve yüz kodlarını algıla
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Algılanan yüzler ile bilinen yüzleri karşılaştırıyoruz
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Bilinmiyor"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Yüzün etrafına bir çerçeve çizelim
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # İsmi yüzün altında gösterelim
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Sonucu gösterelim
    cv2.imshow('Yüz Tanıma', frame)

    # 'q' tuşuna basarak çıkış yapabilirsiniz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Her şey tamamlandığında temizleyelim
video_capture.release()
cv2.destroyAllWindows()
import cv2
import face_recognition

# Video kaynağı olarak kamerayı kullanmak için
video_capture = cv2.VideoCapture(0)

# Tanıtacağımız yüzün resmini yüklüyoruz ve yüz kodunu çıkarıyoruz
known_image = face_recognition.load_image_file("taninacak_yuz.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Bilinen yüzleri ve isimleri listede saklıyoruz
known_face_encodings = [known_face_encoding]
known_face_names = ["Kullanıcı Adı"]

while True:
    # Video akışından bir kare alıyoruz
    ret, frame = video_capture.read()

    # Her kareyi BGR (OpenCV) formatından RGB (face_recognition) formatına dönüştürüyoruz
    rgb_frame = frame[:, :, ::-1]

    # Görüntüdeki tüm yüzleri ve yüz kodlarını algıla
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Algılanan yüzler ile bilinen yüzleri karşılaştırıyoruz
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Bilinmiyor"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Yüzün etrafına bir çerçeve çizelim
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # İsmi yüzün altında gösterelim
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Sonucu gösterelim
    cv2.imshow('Yüz Tanıma', frame)

    # 'q' tuşuna basarak çıkış yapabilirsiniz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Her şey tamamlandığında temizleyelim
video_capture.release()
cv2.destroyAllWindows()
