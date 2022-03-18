#pragma once

#if defined(IMMVISION_USE_GLAD)
    #include <glad/glad.h>
#elif defined(IMMVISION_USE_GLES3)
    #if defined(IOS)
        #include <OpenGLES/ES3/gl.h>
        #include <OpenGLES/ES3/glext.h>
    #elif defined(__EMSCRIPTEN__)
        #include <GLES3/gl3.h>
        #include <GLES3/gl2ext.h>
    #else
        #include <GLES3/gl3.h>
        #include <GLES3/gl3ext.h>
    #endif
#elif defined(IMMVISION_USE_GLES2)
    #ifdef IOS
        #include <OpenGLES/ES2/gl.h>
        #include <OpenGLES/ES2/glext.h>
    #else
        #include <GLES2/gl2.h>
        #include <GLES2/gl2ext.h>
    #endif
#else
    #error "immvision_include_opengl: cannot determine GL include path"
#endif
