cmake_minimum_required(VERSION 3.10)
project(rex_omni_ros)

if("$ENV{ROS_VERSION}" STREQUAL "1")
  find_package(catkin REQUIRED)

  catkin_python_setup()

  catkin_package()

  catkin_install_python(PROGRAMS scripts/rex_omni_server
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(DIRECTORY launch config
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
else()
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_python REQUIRED)

  ament_python_install_package(rex_omni_ros PACKAGE_DIR src/rex_omni_ros)

  install(PROGRAMS scripts/rex_omni_server
    DESTINATION lib/${PROJECT_NAME}
  )

  install(DIRECTORY launch config
    DESTINATION share/${PROJECT_NAME}
  )

  ament_package()
endif()
