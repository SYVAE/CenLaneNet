TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
TARGET =evaluate
DESTDIR =../../
SOURCES += main.cpp \
    src/spline.cpp \
    src/lane_compare.cpp \
    src/counter.cpp

HEADERS += \
    include/spline.hpp \
    include/lane_compare.hpp \
    include/hungarianGraph.hpp \
    include/counter.hpp


INCLUDEPATH +=./include

LIBS+=`pkg-config opencv --cflags --libs`  -lm
