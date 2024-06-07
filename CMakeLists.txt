cmake_minimum_required(VERSION 3.10)
project(practiceRosatom)

# Установите стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Найдите и подключите OpenCV
find_package(OpenCV REQUIRED)



# Укажите путь к заголовочным файлам Eigen (замените на путь к вашей директории)
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/Eigen")

# Добавьте исполняемый файл
add_executable(practiceRosatom
    src/main.cpp
  )

# Укажите, что необходимо использовать директории заголовочных файлов
target_include_directories(practiceRosatom PRIVATE ${OpenCV_INCLUDE_DIRS})

# Свяжите библиотеки с вашим проектом
target_link_libraries(practiceRosatom PRIVATE ${OpenCV_LIBS})

# Подключите Eigen к вашему проекту
target_include_directories(practiceRosatom PRIVATE ${EIGEN3_INCLUDE_DIR})