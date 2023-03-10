#version 410 core

in vec2 UV;

out vec3 color;

uniform sampler2D texIn;

void main(){
	color = texture(texIn, UV).xyz ;
	// color = vec3(1.0f, 1.0f, 0.0f);
}