# HandPose

[WORK IN PROGRESS] *(See the TODO list below)*

A program to recognize hand pose from an RGB camera.

![testing different poses](https://raw.githubusercontent.com/MrEliptik/HandPose/tree/master/Results/result.gif)

## Getting Started

These instructions will help you setting up the project and understanding how the software is working. You'll see the file structure and what each file does. 

### Note

For now, the multithreading only seems to work on windows. I might have a problem of compatibilities between the keras version between Windows and Linux. I'll work on that.

### Requirements

See the *requirements.txt* file or simply run:

    pip install -r requirements.txt

### File structure
.  
├── cnn &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Contains the cnn architecture and the models.**  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── models              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Trained models.**  
├── hand_inference_graph  
├── model-checkpoint  
├── Poses &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**The poses dataset. Each pose will have its folder.**  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Dang  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Fist  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Four  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Palm  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Startrek  
├── protos  
└── utils  

### Running the hand pose recognition

To run the multithreaded hand pose recognition, simply run:

    python HandPose.py


### Adding a new pose

To add a new pose, launch the *AddPose.py* script doing:

    python AddPose.py

You will then be prompted to make a choice. Type "1" and enter. Now you can enter the name of your pose and validate with enter:

    Do you want to :
        1 - Add new pose
        2 - Add examples to existing pose
    1

    Enter a name for the pose you want to add :
    Example

    You'll now be prompted to record the pose you want to add.
                 Please place your hand beforehand facing the
                 camera, and press any key when ready.
                 When finished press 'q'.

Place your hand facing the camera, doing the pose you want to save and press enter when ready. You'll see the camera feed. Move your hand slowly across the frame, closer and further from the camera. Try to rotate a bit your pose. Do every movement slowly as you want to create ghosting.  
You can record for as long as you want, but remember that *camera_fps x seconds_of_recording* images will be generated.   
See an example below:


![recording startrek pose](https://raw.githubusercontent.com/MrEliptik/HandPose/tree/master/Examples/startrek.gif)

Then you want to head to the new pose folder situated in *Poses/name_of_your_pose/name_of_your_pose_1* and manually delete images that doesn't show well your hand pose.   

*You can optionnally bulk rename them once you finished cleaning but note that it's not required.*

Once that is done you want to normalize those newly created images. Launch *normalize.py* with:

    python normalize.py

This script will go to the poses folder and make sure every images is the right size. It will skip those that are already 28x28. 

You then have to retrain the network. For that, open the file situated in "*cnn/cnn.py*" and edit the hyperparameters and the model file name if needed. The saved model will be situated in "*cnn/models/*"

You don't have to specifiy the number of classes, it will be infered from the number of directories under "*Poses/*" .

Launch the training with:

    python cnn/cnn.py


## TODO
- Understand why multithreading doesn't work on linux
- See if Keras is the right version (Windows and Linux)
- ~~Fix multi-threaded detection~~
- ~~Add more examples to each gesture~~
- ~~Add interface to live see inference from network~~
- ~~Test model~~
- ~~Tweak training/structure of CNN~~

## Author

* **Victor MEUNIER** - *HandPose* - [MrEliptik](https://github.com/MrEliptik)

### References

* **Victor Dibia** - *Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow, (2017)*  
GitHub repository, https://github.com/victordibia/handtracking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
