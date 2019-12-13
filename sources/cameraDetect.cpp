
#include "mainFaceSwap.hpp"

// good work
int processCamFaceEyesDetect() {

	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;

	face_cascade.load("data/harr_casecade/haarcascade_frontalface_default.xml");
	eyes_cascade.load("data/harr_casecade/haarcascade_eye.xml");

	// ��ķ ��Ʈ�� �б�
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "<ERROR> Cannot find webcam!" << endl;
		return 0;
	}
	Mat frame;
	while (1) {
		capture.read(frame);
		if (frame.empty()) {
			cout << "<ERROR> No captured frame!" << endl;
			break;
		}
		// �� ���� �� ���
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		std::vector<Rect> faces;
		face_cascade.detectMultiScale(frame_gray, faces);

		for (size_t i = 0; i < faces.size(); i++) {
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
			Mat faceROI = frame_gray(faces[i]);

			// �� ����
			std::vector<Rect> eyes;
			eyes_cascade.detectMultiScale(faceROI, eyes);
			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
				circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
			}

		}

		imshow("ĸ�� - face_cascade and eyes_cascade", frame);

		// ESC Detect
		if (waitKey(10) == 27) {
			break; // escape
		}
	}
	return 0;
}

// good work
int processCamFacemarkDetect() {

	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;
	// Harr cascade �ε�
	face_cascade.load("data/harr_casecade/haarcascade_frontalface_alt2.xml");

	// Facemark�� �ν��Ͻ� ����
	Ptr<Facemark> facemark = FacemarkLBF::create();

	// landmark ������ �ε�
	facemark->loadModel("data/lbfmodel.yaml");

	// ��ķ ��Ʈ�� �б�
	VideoCapture cam(0);
	if (!cam.isOpened()) {
		cout << "<ERROR> Cannot find webcam!" << endl;
		return 0;
	}

	Mat frame, gray;

	// ������ �б�
	while (1) {
		cam.read(frame);
		if (frame.empty()) {
			cout << "<ERROR> No captured frame!" << endl;
			break;
		}

		vector<Rect> faces;
		// �� ������ ���� �׷��̽����� ��ȯ
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// �� ����
		face_cascade.detectMultiScale(gray, faces);

		vector< vector<Point2f> > landmarks;

		// ���帶ũ ����
		bool success = facemark->fit(frame, faces, landmarks);

		if (success)
		{
			// ������
			for (int i = 0; i < landmarks.size(); i++)
			{
				drawLandmarks(frame, landmarks[i]);
			}
		}

		imshow("Facial Landmark Detection", frame);

		if (waitKey(1) == 27) break;

	}

}