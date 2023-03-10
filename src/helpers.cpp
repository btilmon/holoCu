#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "helpers.hpp"

std::string loadStr(std::string filePath)
{
    std::string result;
    std::ifstream loadingStream(filePath, std::ios::in);

    if (loadingStream.is_open()) 
    {
        std::stringstream strStream;
        strStream << loadingStream.rdbuf();
        result = strStream.str();
    }
    else 
        throw std::runtime_error("Could not open file");
    return result;
}

void compileShader(GLuint shaderID, std::string shaderCode, std::string shaderName)
{
    GLint result = GL_FALSE;
    int infoLogLength;

    // std::cout << "Compiling shader: " << shaderName << std::endl;
    char const *shaderCodePointer = shaderCode.c_str();
    glShaderSource(shaderID, 1, &shaderCodePointer, NULL);
    glCompileShader(shaderID);

    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0)
    {
        std::vector<char> errMsg(infoLogLength+1);
        glGetShaderInfoLog(shaderID, infoLogLength, NULL, &errMsg[0]);
        std::cout << &errMsg[0] << std::endl;
    }
}

GLuint loadShader(std::string vsPath, std::string fsPath)
{
    GLuint vsID = glCreateShader(GL_VERTEX_SHADER);
    GLuint fsID = glCreateShader(GL_FRAGMENT_SHADER);

    std::string vsCode = loadStr(vsPath); 
    std::string fsCode = loadStr(fsPath);
    
    compileShader(vsID, vsCode, vsPath);
    compileShader(fsID, fsCode, fsPath);
    
    // std::cout << "Linking the program\n";
    GLuint programID = glCreateProgram();
    glAttachShader(programID, vsID);
    glAttachShader(programID, fsID);
    glLinkProgram(programID);

    GLint result = GL_FALSE;
    int infoLogLength;
    glGetProgramiv(programID, GL_LINK_STATUS, &result);
    glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0)
    {
        std::vector<char> errMsg(infoLogLength+1);
        glGetProgramInfoLog(programID, infoLogLength, NULL, &errMsg[0]);
        // std::cout << &errMsg[0] << std::endl;
    }
    
    glDetachShader(programID, vsID);
    glDetachShader(programID, fsID);

    glDeleteShader(vsID);
    glDeleteShader(fsID);

    return programID;
}