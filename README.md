# 2023 서울드론아카데미 제1기 융합실무 프로젝트
체감형 드론 SW 교육 모델

## MediaPipe 라이브러리
![hand_tracking_3d_android_gpu](https://user-images.githubusercontent.com/81175672/184478283-bec63c44-f298-4c38-b784-ed9409e510a1.gif)                      
[MediaPipe](https://google.github.io/mediapipe/solutions/hands.html) 라이브러리는 구글에서 만들고, 공개, 관리하고 있는 오픈소스 AI 솔루션이다. 해당 라이브러리는 얼굴 인식, 손 인식, 자세 인식 등 여러 데이터를 감지할 수 있는 기능을 제공한다.        

![hand_landmarks](https://user-images.githubusercontent.com/81175672/184479547-361698dd-362a-44c3-9b23-3e6f08ccf179.png)
MediaPipe에서 **손 랜드마크 모델** 은 직접 좌표 예측을 통해 감지된 손 영역 내부의 21개 3D 손 관절 좌표의 정확한 키포인트 위치를 찾아낸다. 이 모델은 일관된 내부 손 포즈 표현을 학습하고 부분적으로 보이는 손과 자체 폐색에도 견고하다.                            

## librosa 라이브러리
![images](https://github.com/RyuJungSoo/Face_Recognition_Elevator/assets/81175672/0947614f-9884-4f4b-88eb-92dcc0fbafe0)                               
**librosa** 는 음악 및 오디오 신호 처리를 위한 파이썬 라이브러리다. 이 라이브러리는 음악 분석, 오디오 신호 변환 및 기타 오디오 처리 작업을 수행하기 위한 다양한 기능을 제공한다.                  


## 개발 환경
**개발 툴** : Google Colab, Visual Studio Code, Anaconda (23.5.2)                                  
**개발 언어** : Python 3.7.16                     
**tensorflow** : 2.10.0                         
**OpenCV** : 4.8.0                     
**MediaPipe** : 0.9.0.1                        
 그 외에 librosa, pytorch, transformers 라이브러리 사용        
                        
드론은 제로 드론을 사용했으며 제로 드론 컨트롤을 위해 **pyirbrain** 라이브러리 사용
***

# 모션 인식
* 데이터 수집 -> 모델 학습 -> 드론 컨트롤
* 시간 순서로 연속된 여러 개의 gesture 데이터를 사용하기 때문에 RNN 중 하나인 LSTM을 사용하여 학습 모델 제작
* LSTM 시퀀스 수 : 30개
* 한 gesture 마다 30초 가량 데이터 수집
* 웹캠으로 모션을 인식시켜 학습한 해당 모션에 대한 명령을 제로 드론으로 전달

# 음성 인식
* librosa와 transformers의 Wav2Vec 라이브러리를 사용
* pyTorch 딥러닝 프레임워크 사용 
* 페이스북에서 사전 학습한 모델 사용
* 입력한 음성 파일을 분석하여 텍스트화 진행
