# 3DAimLab_Vision
Project to explore the computer vision field.

The idea is for the model to perform both random actions (helps with exploration) and model predicted actions to gradually improve the model.

The game is 3D Aim Trainer and the gamemode is Tile Survival.

In order to run this code you need to download interception and python-interception on your own.

Then you should be able to run the code.
The loading code is here for debug purpose if you need to find out what is going wrong, should be easily customizable.

If anything isn't clear open an issue I'll try to help. But this code is here more as an exemple of use for python-interception.



Found a nice idea on how to improve it with post processing of the screenshot to remove ambient noise and unused details.
Should boost performance significantly if it works.

Pre Post Processing :
![my_screenshot0](https://github.com/LjAquinox/3DAimLab_Vision/assets/125894602/bc030764-780d-4b90-b083-aa11d95dabee)

Post Post Precessing :
![my_screenshot0](https://github.com/LjAquinox/3DAimLab_Vision/assets/125894602/a81150d6-8c58-4575-a590-57760f3fb891)


I found a nice way to explore with square bounding box detection thus making saving data with labels way easier.

I first tried my own square Bounding Box detection in a very inefficient way but still interresting (you can find it in FindBoundingBoxesOld.py)
Then I found this [stackoverflow](https://stackoverflow.com/questions/55169645/square-detection-in-image) with a much better way to do it so I adapted it to my use case and it resulted in FindBoundingBoxes.py.

And now I create a new file explore.py that use these BB (bounding boxes) to find where to click. These BB are the future Labels of the image for training.

They look like this :
![screenshot_73](https://github.com/LjAquinox/3DAimLab_Vision/assets/125894602/f74772ea-e6cf-4bf6-9452-84d11f59fa30)

And only take about 1ms to process on my computer. 
With only this explore.py version of the code (no AI involved) I managed to rank 4th.

![image](https://github.com/LjAquinox/3DAimLab_Vision/assets/125894602/56bf35ff-a344-4b3a-9beb-0237890934ae)

The next step is to AI that thing. but I first need to learn more about the YOLO and darknet architecture.

PS : my screen is 1920x1080 and I use the valorant setting with 1.8 sensitivity : Aim and 103 FOV
If this is not your case you need to change the logic of the following functions inside FindBoundingBoxes.py :
  - PositionOfBox_RelativeToCenter()
  - mouse_movement_required_xy()

