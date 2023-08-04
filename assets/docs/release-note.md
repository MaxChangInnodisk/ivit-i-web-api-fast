# Release Note

## r1.1.3
1. Fix: Can not choose the USB camera running with another AI task when adding a new AI Task.
2. Fix: Can not run the object detection model because of losing `libyololayer.so` in the `NVIDIA` platform.
3. Fix: Unplugging the USB camera will cause a blank page in the stream page.
4. Fix: Losing the `Edit` button on the stream page when the AI task is in error status.
5. Fix: The user can switch to the Camera tab when uploading the source file.
6. Fix: Can not save the AI task when re-plugin USB camera on the Edit page.
7. Feat: Add '...' when the project name is too long.
8. Feat: Update USB camera information and sort it automatically when the user opens the dropdown list.
9. Feat: Add AI Task uid and name in some error messages.
10. Feat: Add hardware requirement in `README.md`.

## r1.1.2
1. Change the data process to improve the precision of inference.
2. Display the whole name of the accelerator in the drop-down list.
3. Hang in the loading page when editing the classification task.
4. Keep the dialog when uploading the source file. 
5. Fix the problem that the sample model can be removed in nvidia platform.

## r1.1.1 
1. Set nginx `client_max_body_size` to 0, unlimited uploading files.
2. Stop iVIT-I service when executing `uninstall.sh`.
3. Add fullscreen to the stream page by clicking the streaming.
4. Fix the delete hotkey when editing the area.
5. Fixing the error of executing AI task failed after re-plugging USB camera.
6. Add an error message when unplugging USB camera.
7. Change the format of the log file from `File` to `RotatedFile` and add the `logs` folder and timestamp on the log file ( e.g.`ivit-i-230818.log` )

---

## r1.1
1. Launch / Stop AI Tasks
2. Add / Edit/ Delete AI Tasks.
    1. Select Source Features
        1. Support RTSP, V4L2, Image, Video
    2. Select Model Features
        1. Select from exist model
        2. Import new model ( beta version )
        3. Delete model ( beta version )
