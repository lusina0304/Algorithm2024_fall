#version 330 core

in vec2 UV;
out vec4 FragColor;

uniform sampler2D spriteSampler;
uniform float alpha;

void main() {
    vec4 texColor = texture(spriteSampler, UV);
    if (texColor.a < 0.1)
        discard;
    FragColor = vec4(texColor.rgb, texColor.a * alpha);
}
