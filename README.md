# 2023 서울드론아카데미 제1기 융합실무 프로젝트
드론 컨트롤 보편화를 위한 시스템 구축

## MediaPipe 라이브러리
![hand_tracking_3d_android_gpu](https://user-images.githubusercontent.com/81175672/184478283-bec63c44-f298-4c38-b784-ed9409e510a1.gif)                      
[MediaPipe](https://google.github.io/mediapipe/solutions/hands.html) 라이브러리는 구글에서 만들고, 공개, 관리하고 있는 오픈소스 AI 솔루션이다. 해당 라이브러리는 얼굴 인식, 손 인식, 자세 인식 등 여러 데이터를 감지할 수 있는 기능을 제공한다.        

![hand_landmarks](https://user-images.githubusercontent.com/81175672/184479547-361698dd-362a-44c3-9b23-3e6f08ccf179.png)
MediaPipe에서 **손 랜드마크 모델** 은 직접 좌표 예측을 통해 감지된 손 영역 내부의 21개 3D 손 관절 좌표의 정확한 키포인트 위치를 찾아낸다. 이 모델은 일관된 내부 손 포즈 표현을 학습하고 부분적으로 보이는 손과 자체 폐색에도 견고하다. 
***
그 외에 웹캠으로 입력되는 영상 데이터 처리를 위해 **OpenCV**, 인공지능 모델 학습 및 처리를 위해 **tensorflow** 를 사용한다.                   

## 개발 환경
**개발 툴** : Visual Studio Code, Anaconda (23.5.2)                                  
**개발 언어** : Python 3.7.16                     
**tensorflow** : 2.10.0                         
**OpenCV** : 4.8.0                     
**MediaPipe** : 0.9.0.1                        
