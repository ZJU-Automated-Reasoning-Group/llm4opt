#include <clang-c/Index.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum CXChildVisitResult VarDeclAssignment(CXCursor cursor, CXCursor parent, CXClientData client_data) {
    enum CXCursorKind kind = clang_getCursorKind(cursor);
    if (kind == CXCursor_IntegerLiteral) {
        int client_data = 1;
        return CXChildVisit_Break;
    }
    // print_token(cursor);
    return CXChildVisit_Continue;
}

void getStringBeforeLastEqual(char *input) {
    // 找到最后一个 '=' 的位置
    char *lastEqualSign = strrchr(input, '=');

    // 如果找到 '=', 将字符串在此处截断
    if (lastEqualSign != NULL) {
        *lastEqualSign = '\0'; // 用字符串结束符 '\0' 截断
    }
}

int findLastSubstring(const char *a, const char *b) {
    int lastPosition = -1; // 初始化为 -1，表示没有找到
    int aLen = strlen(a);
    int bLen = strlen(b);

    if (bLen == 0 || aLen < bLen) {
        return lastPosition; // 如果 b 是空字符串或者 a 比 b 短，返回 -1
    }

    for (int i = 0; i <= aLen - bLen; i++) {
        if (strncmp(&a[i], b, bLen) == 0) {
            lastPosition = i; // 更新最后找到的位置
        }
    }

    return lastPosition;
}

// 从文件中提取给定 SourceRange 的源代码
char *extractSourceFromRange(CXSourceRange range) {
    CXSourceLocation start = clang_getRangeStart(range);
    CXSourceLocation end = clang_getRangeEnd(range);

    CXFile file;
    unsigned startOffset, endOffset;
    clang_getFileLocation(start, &file, NULL, NULL, &startOffset);
    clang_getFileLocation(end, NULL, NULL, NULL, &endOffset);

    if (startOffset >= endOffset) {
        return NULL;
    }

    // 读取文件内容
    CXString fileName = clang_getFileName(file);
    const char *filePath = clang_getCString(fileName);
    FILE *fp = fopen(filePath, "r");
    if (!fp) {
        clang_disposeString(fileName);
        return NULL;
    }

    char *buffer = malloc(endOffset - startOffset + 1);
    fseek(fp, startOffset, SEEK_SET);
    fread(buffer, 1, endOffset - startOffset, fp);
    buffer[endOffset - startOffset] = '\0';
    fclose(fp);
    clang_disposeString(fileName);

    return buffer;
}

// 查找变量名在源代码中的实际起始和结束位置
void findVariableLocationInSource(const char *source, const char *varName, unsigned startColumn, unsigned *endColumn, int assign) {
    char *new_source = malloc(strlen(source) + 1); // +1 是为了 null 终止符
    if (new_source == NULL) {
        perror("malloc failed");
        return;
    }

    strcpy(new_source, source);
    if (assign) {
        getStringBeforeLastEqual(new_source);
    }
    int offset = findLastSubstring(new_source, varName);
    if (offset != -1) {
        // printf("source code: %s\n", source);
        // printf("varName: %s\n", varName);
        // printf("offset: %d\n", offset);
        // printf("baseColumn: %d\n", baseColumn);
        *endColumn = startColumn + offset + strlen(varName);
    }
    // 释放内存
    free(new_source);
}

void print_function_info(CXCursor cursor) {

    // Retrieve function name
    CXString function_name = clang_getCursorSpelling(cursor);

    // Retrieve function location
    CXSourceLocation location = clang_getCursorLocation(cursor);
    CXFile file;
    unsigned line, column, offset;
    clang_getFileLocation(location, &file, &line, &column, &offset);

    // Retrieve function type
    CXType function_type = clang_getCursorResultType(cursor);
    CXString type_spelling = clang_getTypeSpelling(function_type);

    // Print function information
    CXCursor definitionCursor = clang_getCursorDefinition(cursor);
    const char *functionType = clang_equalCursors(cursor, definitionCursor) ? "Definition" : "Declaration";
    printf("%s>> %s; Type>> %s; Filename>> %s; Line>> %u; Column>> %u; Parameter list>> (", functionType, clang_getCString(function_name), clang_getCString(type_spelling), clang_getCString(clang_getFileName(file)), line, column);

    int numArgs = clang_Cursor_getNumArguments(cursor);
    for (int i = 0; i < numArgs; i++) {
        CXCursor argCursor = clang_Cursor_getArgument(cursor, i);
        CXType argType = clang_getCursorType(argCursor);
        if (i == 0) {
            printf("%s", clang_getCString(clang_getTypeSpelling(argType)));
        }
        else {
            printf(", %s", clang_getCString(clang_getTypeSpelling(argType)));
        }
    }
    printf(")\n");

    // Dispose of CXStrings
    clang_disposeString(function_name);
    clang_disposeString(type_spelling);
}

void print_varaible_info(CXCursor cursor, CXCursor parent) {
    int assign = 0;
    clang_visitChildren(cursor, VarDeclAssignment, &assign);
    CXString var = clang_getCursorSpelling(cursor);
    const char *name = clang_getCString(var);
    CXSourceRange range = clang_getCursorExtent(cursor);
    CXSourceLocation location = clang_getRangeStart(range);
    CXSourceLocation endLocation = clang_getRangeEnd(range);

    // 获取起始行和列号
    unsigned line, column, offset;
    unsigned endline, endcolumn, endoffset;
    CXFile file, endfile;
    clang_getSpellingLocation(location, &file, &line, &column, &offset);
    clang_getSpellingLocation(endLocation, &endfile, &endline, &endcolumn, &endoffset);

    CXString filename = clang_getFileName(file);
    CXType var_type = clang_getCursorType(cursor);
    CXString type_spelling = clang_getTypeSpelling(var_type);

    // 提取范围内的源代码
    if (clang_getCursorKind(cursor) == CXCursor_VarDecl){
        char *source = extractSourceFromRange(range);
        if (source) {
            // 查找变量名在源代码中的实际范围
            findVariableLocationInSource(source, name, column, &endcolumn, assign);
            free(source);
        }
    }

    // 判断变量是全局变量还是局部变量
    enum CXCursorKind parent_kind = clang_getCursorKind(parent);
    const char *scope = "";
    CXString scope_name = {"", 0};

    // 如果父节点是 TranslationUnit，则是全局变量；否则是局部变量
    if (parent_kind == CXCursor_TranslationUnit) {
        scope = "Global";

        // 打印变量名、作用域、类型及其所在文件中的位置
        printf("Variable>> %s; Type>> %s; Scope>> %s; Filename>> %s; Line>> %u; Column>> %u; endColumn>> %u\n",
            name, clang_getCString(type_spelling), scope, clang_getCString(filename), line, column, endcolumn);

    } else {
        scope = "Local";
        // 获取局部变量的作用域（所在的函数或其他结构）
        CXCursor semantic_parent = clang_getCursorSemanticParent(cursor);
        if (clang_getCursorKind(semantic_parent) == CXCursor_FunctionDecl) {
            scope_name = clang_getCursorSpelling(semantic_parent);  // 获取函数名
        }
        // 打印变量名、作用域、类型及其所在文件中的位置
        printf("Variable>> %s; Type>> %s; Scope>> %s (in %s); Filename>> %s; Line>> %u; Column>> %u; endColumn>> %u\n",
            name, clang_getCString(type_spelling), scope,
            clang_getCString(scope_name), clang_getCString(filename), line, column, endcolumn);
    }

    // 释放 Clang API 分配的内存
    clang_disposeString(var);
    clang_disposeString(filename);
    clang_disposeString(type_spelling);
    if (scope_name.data) {
        clang_disposeString(scope_name);
    }
}

enum CXChildVisitResult fieldVisitor(CXCursor fieldCursor, CXCursor parent, CXClientData client_data) {
    CXString parent_name = clang_getCursorSpelling(parent);
    if (clang_getCursorKind(fieldCursor) == CXCursor_FieldDecl) {
        CXString fieldName = clang_getCursorSpelling(fieldCursor);
        CXType fieldType = clang_getCursorType(fieldCursor);
        // printf("Field Name: %s\n", clang_getCString(fieldName));
        // printf("Field Type: %s\n", clang_getCString(clang_getTypeSpelling(fieldType)));

        CXSourceRange range = clang_getCursorExtent(fieldCursor);
        CXSourceLocation startLocation = clang_getRangeStart(range);
        CXSourceLocation endLocation = clang_getRangeEnd(range);

        CXFile startFile, endFile;
        unsigned int startLine, startColumn, startOffset;
        unsigned int endLine, endColumn, endOffset;

        clang_getFileLocation(startLocation, &startFile, &startLine, &startColumn, &startOffset);
        clang_getFileLocation(endLocation, &endFile, &endLine, &endColumn, &endOffset);
        // printf("Start Location - Line: %u, Column: %u\n", startLine, startColumn);
        // printf("End Location - Line: %u, Column: %u\n", endLine, endColumn);

        printf("Struct>> %s; Type>> %s; Name>> %s; Filename>> %s; Line>> %u; Column>> %u; endColumn>> %u\n", 
            clang_getCString(parent_name), clang_getCString(clang_getTypeSpelling(fieldType)), clang_getCString(fieldName), clang_getCString(clang_getFileName(startFile)), startLine, startColumn, endColumn);
        // printf("Variable: %s; Type: %s; Scope: Local (in Struct %s); Line: %u; Column: %u; endColumn: %u\n", 
        //     clang_getCString(fieldName), clang_getCString(clang_getTypeSpelling(fieldType)), clang_getCString(parent_name), startLine, startColumn, endColumn);
        clang_disposeString(fieldName);
    }
    clang_disposeString(parent_name);
    return CXChildVisit_Continue;
}

void print_struct_info(CXCursor cursor) {
    CXString structName = clang_getCursorSpelling(cursor);
    // CXSourceRange range = clang_getCursorExtent(cursor);
    // CXSourceLocation startLocation = clang_getRangeStart(range);
    // CXSourceLocation endLocation = clang_getRangeEnd(range);

    // unsigned line, column, offset;
    // unsigned endline, endcolumn, endoffset;
    // CXFile file, endfile;

    // clang_getFileLocation(startLocation, &file, &line, &column, &offset);
    // clang_getFileLocation(endLocation, &endfile, &endline, &endcolumn, &endoffset);
    // CXString filename = clang_getFileName(file);

    // printf("Variable> %s; Type> Struct; Filename> %s; Line> %u; endLine> %u; Column> %u; endColumn> %u\n",
    //         clang_getCString(structName), clang_getCString(filename), line, endline, column, endcolumn);

    clang_visitChildren(cursor, fieldVisitor, NULL);

    clang_disposeString(structName);
}

enum CXChildVisitResult visit_node(CXCursor cursor, CXCursor parent, CXClientData client_data) {
    // Check if the cursor represents a function or method declaration
    enum CXCursorKind kind = clang_getCursorKind(cursor);
    if (kind == CXCursor_FunctionDecl || kind == CXCursor_CXXMethod) {
        print_function_info(cursor);
    }
    if (kind == CXCursor_VarDecl || kind == CXCursor_ParmDecl) {
        print_varaible_info(cursor, parent);
    }
    if (kind == CXCursor_StructDecl) {
        print_struct_info(cursor);
    }
    return CXChildVisit_Recurse;
}

void parse_file(const char* filename) {
    // Create translation unit from file
    CXIndex index = clang_createIndex(0, 0);
    CXTranslationUnit translation_unit = clang_parseTranslationUnit(index, filename, NULL, 0, NULL, 0, CXTranslationUnit_None);

    // Check for parsing errors
    if (translation_unit == NULL) {
        fprintf(stderr, "Unable to parse translation unit. Quitting.\n");
        exit(-1);
    }

    // Get translation unit cursor
    CXCursor cursor = clang_getTranslationUnitCursor(translation_unit);

    // Visit all children (functions) in the translation unit
    clang_visitChildren(cursor, visit_node, NULL);

    // Dispose of translation unit and index
    clang_disposeTranslationUnit(translation_unit);
    clang_disposeIndex(index);
}

// clang-16 main.c -L/usr/lib/llvm-16/lib -lclang -I/usr/lib/llvm-16/include -o toolc
int main(int argc, char** argv) {
    // Check for correct usage
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return -1;
    }

    // Parse and analyze the given C file
    // printf("{");
    parse_file(argv[1]);
    // printf("}");

    return 0;
}
