cmake_minimum_required(VERSION 3.10)
project(rex_omni_msgs)

set(MSG_FILES
  BoundingBox.msg
  Point2D.msg
  Polygon2D.msg
  Detection.msg
  PointDetection.msg
  TextRegion.msg
  Keypoint.msg
  KeypointInstance.msg
)

set(SRV_FILES
  Detect.srv
  Point.srv
  DetectWithVisualPrompt.srv
  DetectKeypoints.srv
  RecognizeText.srv
)

if("$ENV{ROS_VERSION}" STREQUAL "1")
  find_package(catkin REQUIRED COMPONENTS
    message_generation
    std_msgs
    sensor_msgs
  )

  add_message_files(FILES ${MSG_FILES})
  add_service_files(FILES ${SRV_FILES})

  generate_messages(DEPENDENCIES std_msgs sensor_msgs)

  catkin_package(CATKIN_DEPENDS message_runtime std_msgs sensor_msgs)
else()
  find_package(ament_cmake REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)

  set(INTERFACE_FILES)
  foreach(file ${MSG_FILES})
    list(APPEND INTERFACE_FILES "msg/${file}")
  endforeach()
  foreach(file ${SRV_FILES})
    list(APPEND INTERFACE_FILES "srv/${file}")
  endforeach()

  rosidl_generate_interfaces(${PROJECT_NAME}
    ${INTERFACE_FILES}
    DEPENDENCIES std_msgs sensor_msgs
  )

  ament_export_dependencies(rosidl_default_runtime)
  ament_package()
endif()
