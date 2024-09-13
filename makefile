CXXFLAGS = -I /opt/homebrew/include/eigen3
CXX = g++ -std=c++20

TARGET = main
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)