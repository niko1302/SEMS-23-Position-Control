# SEMS-23-Depth_Control

## Requirements
The Custom Messages Package is required as a Node:

```
https://github.com/niko1302/SEMS-23-Custom_Messages.git 
```

## Launch the Node

To Start the Node Type:

```
ros2 launch depth_control depth.launch.py vehicle_name:=<Vehicle Name>
```

The Launch Argument 'vehicle_name' has no deafult value and thus has to be set in the Terminal.

## Parameters

### Change and List Paramters

To list all current Parameters in the Terminal type:
```
ros2 param list
```

Change a parameters default value, by editing the `.yaml` files in the `config/` folder.
To change the parameters during runtime enter:
```
rqt
```
and navigate to "Plugins" -> "Configuration" -> "Dynamic Reconfigure"

