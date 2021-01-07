# MP_FairFace: Assignment to join Professor Joo's Research

Original paper: https://arxiv.org/abs/1908.04913

Original GitHub Repo: https://github.com/dchen236/FairFace 

Goal: To try to replicate the results from the paper 'FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age', as well as learn more about Professor Joo's research.

### Progress

- Reviewed paper ‘FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age’
- Reviewed Professor Joo's past research through his website and Google Scholar
- Experimentation with given data and model
	- Setup local environment, ran pretrained models on test data, verified results
	 	- Ran pretrained models on given test images as well as my own test images
- Working on training my own model to replicate results 
	- Using dataset from original GitHub repo
	- Currently running into issues (see below section)


### Issues

- Training the model would take days/weeks to complete at the current rate - Am I doing something wrong?
	- Tried running code both locally with Jupyter Notebook and with Google colab - wondering if the issue with my hardware or approach?
		- Approach: Using the data given in the original GitHub repo, same model architecture as specified in the paper
			- Model Architecture:
				- torchvision.models ResNet-34(pretrained=False)
				- Criterion torch.nn.CrossEntropyLoss
				- Optimizer torch.optim.Adam
	- Misc issues with Google Colab
		- In Google Colab, ran into issue: "OSError: [Errno 5] Input/output error," which is a known issue when files are too large
		- In Google Colab, train dataset would be corrupted (missing files) when uploaded to drive
			- Wrote script to catch corruptions to fix this
	- Currently I have no access to GPU or CUDA

- How should I verify the accuracy of the model, given that I do not have access to the test datasets descirbed in the paper?
	- Was planning to use the given val dataset as the test set, then split the train dataset into train/val to get high-level idea of model performance
		- Ran into issue above with training
	- Test datasets mentioned in paper: Geo-tagged Tweets, Media Photographs, Protest Dataset


### Ongoing Research I am Interested In

- FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age
- Cultural Diffusion and Trends in Facebook Photographs
- Understanding the Political Ideology of Legislators from Social Media Images
- Protest Activity Detection and Perceived Violence Estimation from Social Media Images


### Relevant Files

- MP_FairFace_Model_Training.ipynb = Model Training code
- Check_Train_Folder_Contents_Script.ipynb = Script to check for data corruptions in Google Drive
- Note: The datasets, as well as pretrained models are not included in this repo, but can be found in the Original GitHub Repo above. 
	- They are not included as they are either exceed GitHub's max file size or it took too long to upload

