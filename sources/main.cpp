/*
Problems)
Fianl project
- 두 장의 인물 사진이 주어지면 두 장의 얼굴을 서로 바꾼 이미지를 만든다
- OpenCV의 얼굴인식 API를 사용해서 얼굴 부분을 인식해 낼 것
- 헤어스타일은 변화하지 않고 눈, 코, 입 등 얼굴 부분만 교체


Written by cybertramp(paran_son@outlook.com)

*/

#include "mainFaceSwap.hpp"

bool debug_option = false;

/* 
 * 랜드마크 포인트 선 그리기 함수 
 */
void drawPolyline(
	Mat& img,
	const vector<Point2f>& landmarks,
	const int start,
	const int end,
	bool isClosed = false
){
	// 시작점과 끝점들을 points에 넣음
	vector <Point> points;
	for (int i = start; i <= end; i++)
	{
		points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
	}
	// 폴리라인 그리기
	polylines(img, points, isClosed, Scalar(0, 100, 255), 2, 16);
}

/*
 * 랜드마크 포인트 그리기
 */
void drawLandmarksPoints(Mat& im, vector<Point2f>& landmarks){
	// 점 그리기
	for (int i = 0; i < 68; i++)
		line(im, landmarks[i], landmarks[i], Scalar(0, 0, 255), 3);
}

/* 
 * 얼굴 랜드마크 그리기 
 */
void drawLandmarks(Mat& im, vector<Point2f>& landmarks){	
	// 68 점을 통해 얼굴을 그림
	if (landmarks.size() == 68)
	{
		drawPolyline(im, landmarks, 0, 16);           // 턱 라인
		drawPolyline(im, landmarks, 17, 21);          // 왼쪽 눈섭
		drawPolyline(im, landmarks, 22, 26);          // 오른쪽 눈섭
		drawPolyline(im, landmarks, 27, 30);          // 코 가운데 라인
		drawPolyline(im, landmarks, 30, 35, true);    // 코 사이드
		drawPolyline(im, landmarks, 36, 41, true);    // 왼쪽 눈
		drawPolyline(im, landmarks, 42, 47, true);    // 오른쪽 눈
		drawPolyline(im, landmarks, 48, 59, true);    // 입술 바깥
		drawPolyline(im, landmarks, 60, 67, true);    // 입술 안
	}
	else
	{	// 점의 수가 68개가 아닌 경우, 어떤 점이 얼굴 특징에 해당하는지 알수 없기에
		// landmark 당 하나의 점을 그림
		for (int i = 0; i < landmarks.size(); i++)
		{
			circle(im, landmarks[i], 3, Scalar(255, 100, 0), FILLED);
		}
	}

}

/* 
 * 이미지에서 얼굴을 감지하여 vector<vector<Point2f>>로 반환 
 */
vector<Point2f> imgFaceDetection(Mat img, String title, CascadeClassifier faceCascade, Ptr<Facemark> faceMark) {

	Mat frame, gray;
	frame=img.clone();
	
	vector<Rect> faces;

	// facedetector를 위해 GRAY로 변환
	cvtColor(frame, gray, COLOR_BGR2GRAY);

	// 얼굴 감지
	faceCascade.detectMultiScale(gray, faces);

	// 감지한 얼굴 포인트들
	vector<vector<Point2f>> landmarks;
	vector<Point2f> points;

	// landmark 감지
	bool success = faceMark->fit(frame, faces, landmarks);
	
	if (success){	// 성공시
		
		if (debug_option) {
			drawLandmarksPoints(frame, landmarks[0]);
			imshow("이미지 - 랜드마크", frame);
			drawLandmarks(frame, landmarks[0]);
			imshow("이미지 - 랜드마크", frame);
			cout << landmarks[0] << endl;

		}

		for (int i = 0; i < landmarks[0].size(); i++) {
			points.push_back(Point2f(landmarks[0][i].x, landmarks[0][i].y));
		}
	}
	else {
		cout << "<ERROR> landmark 감지 실패" << endl;
	}
	if(debug_option)
		imshow(title, frame);
	
	return points;
}

/*
 * 들로네 삼각 분할(반환: 각 삼각형의 3점의 벡터)
 */
static void calcDelaunayTriangles(Rect rect, vector<Point2f>& points, vector< vector<int> >& delaunayTri, Mat img) {

	Mat img_tmp = img.clone();
	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// Insert points into subdiv
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
		subdiv.insert(*it);

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point2f> pt(3);
	vector<int> ind(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			for (int j = 0; j < 3; j++)
				for (size_t k = 0; k < points.size(); k++)
					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = k;
			if (debug_option) {
				// DelaunayTriangles 그리기
				line(img_tmp, pt[0], pt[1], (255, 255, 255), 1, 16, 0);
				line(img_tmp, pt[1], pt[2], (255, 255, 255), 1, 16, 0);
				line(img_tmp, pt[2], pt[0], (255, 255, 255), 1, 16, 0);
			}
			

			delaunayTri.push_back(ind);
		}
	}
	if(debug_option)
		imshow("이미지2 - 델루나이 삼각형", img_tmp);
}

/*
* 워프 및 알파 - img1과 img2에서 img까지 삼각 영역을 혼합
*/
void warpTriangle(Mat& img1, Mat& img2, vector<Point2f>& t1, vector<Point2f>& t2){

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	// 각 사각형의 왼쪽 상단 모서리를 기준으로 오프셋 지점
	vector<Point2f> t1Rect, t2Rect;
	vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{
		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly
	}

	// 삼각형을 채워 mask 가져오기
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
	
	// 작은 직사각형 패치들에 warpimage 적용
	Mat img1Rect;
	img1(r1).copyTo(img1Rect);
	
	Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

	//srcTri 와 dstTri를 사용하여 계산된 어파인 변환을 src에 적용
	Mat warpMat = getAffineTransform(t1Rect, t2Rect);
	warpAffine(img1Rect, img2Rect, warpMat, img2Rect.size(), INTER_LINEAR, BORDER_REFLECT_101);

	multiply(img2Rect, mask, img2Rect);
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;
}
/*
* 얼굴 스왑 처리
*/
void processFaceSwap(Mat &img1, Mat &img2, Mat &res) {
	
	Mat img1_ori = img1.clone();

	string windowsTitle1 = "이미지1 - 얼굴감지";
	string windowsTitle2 = "이미지2 - 얼굴감지";

	imshow("이미지1 - 얼굴 적용할 대상", img1);
	imshow("이미지2 - 얼굴이 바뀔 사진", img2);
	//// 1. Face Alignment
	//// 1-1. Facial Landmark Detection

	// 얼굴 탐지를 위한 모델 미리 불러오기
	CascadeClassifier face_cascade;

	// Harr casecade 로드
	face_cascade.load("data/harr_casecade/haarcascade_frontalface_alt2.xml");

	// Facemark 인스턴스 생성
	Ptr<Facemark> facemark = FacemarkLBF::create();

	// landmark 감지 모델 로드
	facemark->loadModel("data/lbfmodel.yaml");

	// face points 얻기
	vector<Point2f> face_points1 = imgFaceDetection(img1, windowsTitle1, face_cascade,facemark);
	vector<Point2f> face_points2 = imgFaceDetection(img2, windowsTitle2, face_cascade, facemark);

	// CV_32F로 변환
	img1.convertTo(img1, CV_32F);
	res.convertTo(res, CV_32F);
	
	//// 1-2. Find Convex Hull
	// convex hull 찾기 - 외곽선 찾기 알고리즘
	vector<Point2f> hull1;
	vector<Point2f> hull2;
	vector<int> hullIndex;

	convexHull(face_points2, hullIndex, false, false);

	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(face_points1[hullIndex[i]]);
		hull2.push_back(face_points2[hullIndex[i]]);
	}
	//// 1-3. Delaunay Triangulation
	// convex hull을 통해 delaunay triangulation 찾기
	vector< vector<int> > dt;
	Rect rect(0, 0,res.cols, res.rows);		// 결과 담을 사각형

	calcDelaunayTriangles(rect, hull2, dt,img2);

	//// 1-4. Affine warp triangles
	// Delaunay 삼각형들에 affine 변환을 적용

	for (size_t i = 0; i < dt.size(); i++)
	{
		vector<Point2f> t1, t2;
		// 삼격형들에 해당하는 img1, img2에 대한 점을 구함
		for (size_t j = 0; j < 3; j++)
		{
			t1.push_back(hull1[dt[i][j]]);
			t2.push_back(hull2[dt[i][j]]);

		}
		
		warpTriangle(img1, res, t1, t2);
		if(debug_option){
			line(img1_ori, hull1[dt[i][0]], hull1[dt[i][1]], (255, 255, 255), 1, 16, 0);
			line(img1_ori, hull1[dt[i][1]], hull1[dt[i][2]], (255, 255, 255), 1, 16, 0);
			line(img1_ori, hull1[dt[i][2]], hull1[dt[i][0]], (255, 255, 255), 1, 16, 0);
		}
		
	}
	if(debug_option)
		imshow("이미지1 - 이미지2의 들로니 삼각형 어파인 적용", img1_ori);
	
	// mask 계산
	vector<Point> hull8U;
	for (int i = 0; i < hull2.size(); i++)
	{
		Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}

	Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255));
	if (debug_option) {
		imshow("이미지2 - 마스크", mask);
	}
	
	//// 2. Seamless Cloning
	Rect r = boundingRect(hull2);  // img2에 대한 convex hull을 사각형으로
	Point center = (r.tl() + r.br()) / 2;  // img2에 대한 convex hull의 중심점을 찾음

	Mat img_seamlessCloned;
	res.convertTo(res, CV_8UC3);
	if(debug_option)
		imshow("이미지2 - seamlessClone 전", res);
	seamlessClone(res, img2, mask, center, img_seamlessCloned, NORMAL_CLONE);

	res = img_seamlessCloned;

	imshow("이미지3 - 결과", res);
}

int main(int argc, char** argv) {

	// console set
	HWND console = GetConsoleWindow();
	RECT ConsoleRect;
	GetWindowRect(console, &ConsoleRect);
	SetWindowPos(console, 0, 500, 500, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
	MoveWindow(console, ConsoleRect.left, ConsoleRect.top, 640, 320, TRUE);
	SetWindowTextA(console,"FaceSwap Program");

	// 얼굴 스왑할 이미지 파일명
	string image1Filename = "images/img1.jpg";
	string image2Filename = "images/img2.jpg";
	string image3Filename = "images/res.jpg";

	Mat imgage1, imgage2, imgage3;

	int input = 99;

	cout << "=========================" << endl;
	cout << "| [Face Swap Program]" << endl;
	cout << "| 1. Faceswap from images" << endl;
	cout << "| 2. Faceswap from images(DEBUG MODE)" << endl;
	cout << "| 3. FaceEyes Detect from Webcam" << endl;
	cout << "| 4. Facemark Detect from Webcam" << endl;
	cout << "| 0. Exit" << endl;
	cout << "=========================" <<endl;
	cout << "Input) ";
	cin >> input;

	// 파일 로드
	imgage1 = imread(image1Filename);
	imgage2 = imread(image2Filename);
	imgage3 = imgage2.clone();
	if (imgage1.empty() || imgage2.empty()) {
		cout << "<ERROR> Cannot read Image file." << endl;
		return 0;
	}

	switch (input) {
		case 1:
			// 얼굴바꾸기 처리
			debug_option = FALSE;
			processFaceSwap(imgage1, imgage2, imgage3);
			imwrite(image3Filename, imgage3);
			waitKey(0);
			break;
		case 2:
			// DEBUG MODE
			debug_option = TRUE;
			processFaceSwap(imgage1, imgage2, imgage3);
			imwrite(image3Filename, imgage3);
			waitKey(0);
			break;
		case 3:
			processCamFaceEyesDetect();
			break;
		case 4:
			processCamFacemarkDetect();
			break;
		case 0:
			break;
	}
	return 0;
}
