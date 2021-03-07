# TAU_DL
Acoustic Scene Classification and Time of Day Estimation



**Abstract**

We present a system for time-of-day prediction based on recorded environmental acoustic information. The system’s architecture is inspired by a recent work in the field of acoustic scene classification, and is comprised of an ensemble model of deep convolutional neural networks trained separately on differently processed versions of the raw audio data. We address the problem of attempting to predict a continuous parameter with a bounded, cyclic nature using machine leaning methodologies, identify some key restrictions this problem entails and propose a solution. The proposed model’s predications achieve a root mean-squared-error of  116.02 minutes on the evaluation data set. Finally, we utilize some recently proposed methods to try and gain intuition into the trained model’s decision making process, and attempt to identify meaningful features in the processed input data.



**Appendix**

https://github.com/pranjaldatta/Denoising-Autoencoder-in-Pytorch

https://www.kaggle.com/talmanr/cnn-with-pytorch-using-mel-features

https://github.com/kfir1989/TAU_DL




**Results**

ALGORITHMS	VALIDATION ACC.	EVALUATION ACC

MONO (MEL)	0.95  	0.827

HPSS	       0.89	        0.715

ENCODED	0.95    	0.831

ENSEMBLE	0.964	0.836


Future Work

 <>

