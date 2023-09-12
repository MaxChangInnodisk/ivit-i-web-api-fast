# Release Note

## v1.1.6
* Web API
  1. Fixed the incorrect inference FPS.
  2. Fixed disclaimer issue when using docker/install.sh to install service.
  3. Updated application to v1.1.6: refactor and optimized whole applications.

## v1.1.5
* Web API
  1. Fix the version of the micro-service, like rtsp-simple-server, rtsptoweb, and nginx.
  2. Set "draw_bbox" and "draw_result" to True in DetectionZone ( Application ).
  3. Update disclaimer and eula.

## v1.1.4
* Web API
  1. Feat: Check the value of the source when editing the AI task. Avoid writing the empty into the database.
  2. Feat: Avoid SQL injection when editing and deleting the AI task.
  3. Fix: Change the workflow to capture the USB camera to avoid buffer conflict.
  4. Fix: The application ( Detection_Zone ) should draw the correct label name on the left-top corner.

* Website
  1. Feat: Add an error message in the entrance when the Web API is unavailable.
  2. Fix: When editing the AI task that uses the USB camera, the save button will be disabled when unplugging the camera.
  3. Fix: Check whether the source dropdown list is empty when editing the AI task.
  4. Fix: Use the default color palette if the Web API doesn't provide it. 

## v1.1.3
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

## v1.1.2
1. Change the data process to improve the precision of inference.
2. Display the whole name of the accelerator in the drop-down list.
3. Hang in the loading page when editing the classification task.
4. Keep the dialog when uploading the source file. 
5. Fix the problem that the sample model can be removed in nvidia platform.

## v1.1.1 
1. Set nginx `client_max_body_size` to 0, unlimited uploading files.
2. Stop iVIT-I service when executing `uninstall.sh`.
3. Add fullscreen to the stream page by clicking the streaming.
4. Fix the delete hotkey when editing the area.
5. Fixing the error of executing AI task failed after re-plugging USB camera.
6. Add an error message when unplugging USB camera.
7. Change the format of the log file from `File` to `RotatedFile` and add the `logs` folder and timestamp on the log file ( e.g.`ivit-i-230818.log` )

---

## v1.1.0
1. Launch / Stop AI Tasks
2. Add / Edit/ Delete AI Tasks.
    1. Select Source Features
        1. Support RTSP, V4L2, Image, Video
    2. Select Model Features
        1. Select from exist model
        2. Import new model ( beta version )
        3. Delete model ( beta version )
