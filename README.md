# NeurIPS LLM Efficiency Challenge Student Submission
Team name: **ReaLLM Conquerors**

## Description
* Our submissions for the 4090 track are under [4090_submissions](4090_submissions/) folder
    - There are three submissions for this track: [4090_1st_submission](4090_submissions/4090_1st_submission), [4090_2nd_submission](4090_submissions/4090_2nd_submission), and [4090_3rd_submission](4090_submissions/4090_3rd_submission)
* Our submissions for the A100 track are under [A100_submissions](A100_submissions/) folder
    - There are three submissions for this track: [A100_1st_submission](A100_submissions/A100_1st_submission), [A100_2nd_submission](A100_submissions/A100_2nd_submission), and [A100_3rd_submission](A100_submissions/A100_3rd_submission)
 
* Our training code for 4090 track is inside the folder 4090_training_code. Use the dockerfile which will run the training and then the final artifact is present at /submission/qwen_ours_3e-5_4bit/checkpoint-2340 which is an adapter and can be gotten using the docker cp command. We would like to use our first submission

* Our training code for A100 track is inside the folder A100_training_code. Use the dockerfile which will run the training and then the final artifact is present at /submission/qwen_ours_3e-5_new/checkpoint-2340 which is an adapter for qwen and can be gotten using the docker cp command. We would like to use our first submission

## Public Dataset Download Instruction
To download the dataset:
```bash
gdown 1E43WbnyL8iXzOw21ye95VdJAYyra6ewd
```
