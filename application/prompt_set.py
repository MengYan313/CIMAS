idiom_judge_prompt = [
    # 1. Definition and properties of idioms; 2. Three steps for idiom judgment
    {"role": "system", "content": (
        "You are an expert in program synthesis and idiom judgment.\n"
        "Code idioms refer to patterns with both high semantic clarity and commonly used in source code.\n"
        "Code that is constructed in a natural and intuitive way may be regarded as idiomatic.\n"
        "\n"
        "To be considered an idiom, a Code Fragment must satisfy all three of the following steps:\n"
        "<Step 1>: Determine whether the Code Fragment possesses clear semantics and readable variables.\n"
        "<Step 2>: Assess whether the Code Fragment is suitable for common application scenarios.\n"
        "<Step 3>: Determine whether the Code Fragment provides enough information so that its purpose and functionality can be inferred without context.\n"
        "\n"
        "Returns whether the current Code Fragment is a legitimate code idiom and provide reasons based on the above three steps.\n"
        "Below are several example dialogues:"
    )},
    # 3. Example dialogues and reasons for idiom judgment
    # Simple return statement
    {"role": "user", "content": "Determine whether the Code Fragment [return 0;] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "No, the Code Fragment is not a legitimate code idiom. Because it does not satisfy <Step 1> and <Step 3>.\n"
        "<Step 1>: No, it does not possess clear semantics as it is too simple and common.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as indicating successful execution of a function.\n"
        "<Step 3>: No, the [Simple Return Statement] does not contain enough information to infer its purpose and functionality."
    )},
    # Complex return statement
    {"role": "user", "content": "Determine whether the Code Fragment [return (comment && XMLUtil::StringEqual(comment->Value(), Value()))] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "Yes, the Code Fragment is a legitimate code idiom. Because the above three steps are satisfied.\n"
        "<Step 1>: Yes, it possesses clear semantics and readable variables. The variable [comment] is a pointer, and the function [XMLUtil::StringEqual] suggests that it is checking whether two string values are equal, likely in an XML parsing context.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as validating whether a given XML comment matches a specific value, which is a common operation in XML parsing.\n"
        "<Step 3>: Yes, the [Complex Return Value Content] provides sufficient information to infer its purpose and functionality."
    )},
    # Simple assignment statement
    {"role": "user", "content": "Determine whether the Code Fragment [int i = 0;] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "No, the Code Fragment is not a legitimate code idiom. Because it does not satisfy <Step 1> and <Step 3>.\n"
        "<Step 1>: No, it does not possess clear semantics as it is too simple and common. And the variable [i] is not readable.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as initializing a loop counter.\n"
        "<Step 3>: No, the [Simple Assignment Statement] does not contain enough information to infer its purpose and functionality."   
    )},
    # Complex assignment statement
    {"role": "user", "content": "Determine whether the Code Fragment [digit = *q - 'a' + 10] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "Yes, the Code Fragment is a legitimate code idiom. Because the above three steps are satisfied.\n"
        "<Step 1>: Yes, it possesses clear semantics as it converts a character to a digit value. And the variable [digit] is readable.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as parsing hexadecimal digits.\n"
        "<Step 3>: Yes, the [Complex Right Value of the Assignment Statement] provides sufficient information to infer its purpose and functionality."
    )},
    # Complex assignment statement
    {"role": "user", "content": "Determine whether the Code Fragment [const unsigned int digitScaled = mult * digit;] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "Yes, the Code Fragment is a legitimate code idiom. Because the above three steps are satisfied.\n"
        "<Step 1>: Yes, it possesses clear semantics as it scales a digit by a multiplier. And the variable [digit], [mult] and [digitScaled] is readable.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as performing arithmetic operations on digits.\n"
        "<Step 3>: Yes, the [Complex Right Value of the Assignment Statement] provides sufficient information to infer its purpose and functionality."
    )},
    # Simple comparison statement
    {"role": "user", "content": "Determine whether the Code Fragment [p < q] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "No, the Code Fragment is not a legitimate code idiom. Because it does not satisfy <Step 1> and <Step 3>.\n"
        "<Step 1>: No, it does not possess clear semantics as it is too simple and common. And the variable [p] and [q] is not readable.\n"
        "<Step 2>: Yes, it could be suitable for common application scenarios, such as a comparison operation.\n"
        "<Step 3>: No, the [Simple Comparison Expression] does not contain enough information to infer its purpose and functionality."      
    )},
    # Complex comparison statement
    {"role": "user", "content": "Determine whether the Code Fragment [(_flags & NEEDS_NEWLINE_NORMALIZATION) && *p == CR] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "Yes, the Code Fragment is a legitimate code idiom. Because the above three steps are satisfied.\n"
        "<Step 1>: Yes, it possesses clear semantics and readable variables. The bitwise AND operation [_flags & NEEDS_NEWLINE_NORMALIZATION] checks whether a specific flag is set, and the comparison [*p == CR] indicates that it is checking whether the character pointer [p] points to a carriage return character (CR).\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as handling newline normalization in text processing, where CR (Carriage Return) might need to be processed differently.\n"
        "<Step 3>: Yes, the [Multiple Comparison Expressions] provides enough information to infer its purpose and functionality without additional context. The combination of flag checking and character comparison suggests that it is part of a text normalization routine, making it a meaningful and self-contained operation."
    )},
    # Control flow statement
    {"role": "user", "content": "Determine whether the Code Fragment [if ( !doc ) { doc = _document;}] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "Yes, the Code Fragment is a legitimate code idiom. Because the above three steps are satisfied.\n"
        "<Step 1>: Yes, it possesses clear semantics as it checks if [doc] is null and assigns it a value if it is. And the variable [doc] and [_document] is readable.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as lazy initialization or ensuring a variable is assigned.\n"
        "<Step 3>: Yes, the [Conditional Statement and its Content] provides sufficient information to infer its purpose and functionality."
    )},
    # Function call statement
    {"role": "user", "content": "Determine whether the Code Fragment [void XMLUtil::ToStr( int v, char* buffer, int bufferSize ) { TIXML_SNPRINTF( buffer, bufferSize, \"%d\", v );}] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "Yes, the Code Fragment is a legitimate code idiom. Because the above three steps are satisfied.\n"
        "<Step 1>: Yes, it possesses clear semantics as it converts an integer to a string and stores it in a buffer. And the variable [buffer] and [bufferSize] is readable.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as formatting and outputting data.\n"
        "<Step 3>: Yes, the [Function Declaration and its Content] provides sufficient information to infer its purpose and functionality."
    )},
    # File operation statement
    {"role": "user", "content": "Determine whether the Code Fragment [fclose( fp )] is a legitimate code idiom."},
    {"role": "assistant", "content": (
        "Yes, the Code Fragment is a legitimate code idiom. Because the above three steps are satisfied.\n"
        "<Step 1>: Yes, it possesses clear semantics as it closes a file pointer. And the variable [fp] is readable.\n"
        "<Step 2>: Yes, it is suitable for common application scenarios, such as closing a file after operations are done.\n"
        "<Step 3>: Yes, the [Close File Statement] provides sufficient information to infer its purpose and functionality."
    )}
]

idiom_synthesis_prompt = [
    # 1. Definition and properties of idioms; 2. General cases and criteria for idiom synthesis
    {"role": "system", "content": (
        "You are an expert in program synthesis and idiom synthesis.\n"
        "Code idioms refer to patterns with both high semantic clarity and commonly used in source code.\n"
        "Code that is constructed in a natural and intuitive way may be regarded as idiomatic.\n"
        "\n"
        "Determine whether Idiom 1 and Idiom 2 can be synthesized into a reasonable New Idiom.\n"
        "Idiom synthesis is possible if at least one of the following conditions holds:\n"
        "<Feature 1>: Idiom 1 and Idiom 2 share common variables or contain significant overlapping code segments.\n"
        "<Feature 2>: Idiom 1 and Idiom 2 have a strong semantic relationship or exhibit a clear sequential dependency.\n"
        "<Feature 3>: Idiom 1 and Idiom 2 are frequently used together as a common combination or exhibit a clear sequential dependency.\n"
        "\n"
        "If at least one of these conditions is satisfied, synthesize Idiom 1 and Idiom 2 into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics. Only return the final synthesized idiom. Do not provide any additional explanations.\n"
        "If synthesis is not possible, return 'no' and provide reasons based on the above three features.\n"
        "Below are several example dialogues:"
    )},
    # 3. Example dialogues and reasons for idiom synthesis
    # malloc & free combination
    {"role": "user", "content": "Integrate Idiom 1 [char* buffer = (char*)malloc(256); if (buffer == NULL) { fprintf(stderr, \"Memory allocation failed\\n\"); exit(1); }] and Idiom 2 [free(buffer);] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "char* buffer = (char*)malloc(256); if (buffer == NULL) { fprintf(stderr, \"Memory allocation failed\\n\"); exit(1); }\nfree(buffer);"},
    # label & goto combination
    {"role": "user", "content": "Integrate Idiom 1 [label: std::cout << \"This is a label statement.\" << std::endl;] and Idiom 2 [goto label;] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "label: std::cout << \"This is a label statement.\" << std::endl;\ngoto label;"},
    # Common variable q
    {"role": "user", "content": "Integrate Idiom 1 [const char* q = p + 3;] and Idiom 2 [const size_t delta = q - p;] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "const char* q = p + 3;\nconst size_t delta = q - p;"},
    # Common variable m_willBeWrittenTo
    {"role": "user", "content": "Integrate Idiom 1 [bool m_willBeWrittenTo = false;] and Idiom 2 [bool willBeWrittenTo() const{ return m_willBeWrittenTo;}] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "bool m_willBeWrittenTo = false;\nbool willBeWrittenTo() const{ return m_willBeWrittenTo;}"},
    # Common variable smtSort
    {"role": "user", "content": "Integrate Idiom 1 [auto smtSort = [](auto _var){ return smt::smtSortAbstractFunction(*_var->type()); };] and Idiom 2 [auto inputSorts = applyMap(_function.parameters(), smtSort);] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "auto smtSort = [](auto _var){ return smt::smtSortAbstractFunction(*_var->type()); };\nauto inputSorts = applyMap(_function.parameters(), smtSort);"},
    # Semantic association m_inRequire & m_insideModifierInvocation
    {"role": "user", "content": "Integrate Idiom 1 [if (functionType->kind() == FunctionType::Kind::Require) { solAssert(!m_inRequire); m_inRequire = true;}] and Idiom 2 [m_insideModifierInvocation = true;] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "if (functionType->kind() == FunctionType::Kind::Require) { solAssert(!m_inRequire); m_inRequire = true;}\nm_insideModifierInvocation = true;"},
    # Negative example, cannot be synthesized
    {"role": "user", "content": "Integrate Idiom 1 [static const char* xmlHeader = \"<?\";] and Idiom 2 [const char* q = p + 3;] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "No, because the two Code Fragments have neither common variables nor semantic connections or common combinations."},
    # Open file & close file combination
    {"role": "user", "content": "Integrate Idiom 1 [FILE *fp = fopen(\"example.txt\", \"r\");] and Idiom 2 [fclose(fp);] into a reasonable New Idiom. The New Idiom must include Idiom 1 and Idiom 2, and the position of each sentence in the New Idiom must be arranged reasonably to ensure logical coherence and clear semantics."},
    {"role": "assistant", "content": "FILE *fp = fopen(\"example.txt\", \"r\");\nfclose(fp);"}
]

standard_prompt_1 = [
    # 1. Programming standard definition
    {"role": "system", "content": (
        "You are an expert in C++ programming and follow excellent programming standards. One variable-related programming standard is: Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared. Below is a code idiom, please determine whether it reflects the above programming standard.\n"
        "If the given code idiom reflects this programming standard well, return <yes>; if it violates this programming standard, return <no>; if the code idiom is irrelevant to this programming standard, return <else>\n"
        "You should and only should return yes when the following template is satisfied: [Variable Type][Variable Name]=[Initial Value or Function Call Return Value], and the variable is one of pointer variable, resource descriptor variable, or BOOL variable\n"
        "You should and only should return no when the following template is satisfied: [Variable Type][Variable Name]\n"
        "\n"
        "Here are some example dialogues:"
    )},
    # 2. Positive and negative examples for this programming standard
    # Positive example
    {"role": "user", "content": "Code idiom [SOCKET s = INVALID_SOCKET;] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "yes. Satisfies [Variable Type][Variable Name]=[Initial Value], and variable [s] is a resource descriptor variable, and is initialized with [INVALID_SOCKET] on the same line"},
    # Positive example
    {"role": "user", "content": "Code idiom [unsigned char *msg = NULL;] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "yes. Satisfies [Variable Type][Variable Name]=[Initial Value], and variable [msg] is a pointer variable, and is initialized with [NULL] on the same line"},
    # Positive example
    {"role": "user", "content": "Code idiom [BOOL success = FALSE;] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "yes. Satisfies [Variable Type][Variable Name]=[Initial Value], and variable [success] is a BOOL variable, and is initialized with [FALSE] on the same line"},
    # Negative example
    {"role": "user", "content": "Code idiom [SOCKET s;] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "no. Satisfies [Variable Type][Variable Name], and only declares resource descriptor variable [s] without assignment expression '='"},
    # Negative example
    {"role": "user", "content": "Code idiom [unsigned char *msg;] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "no. Satisfies [Variable Type][Variable Name], and only declares pointer variable [msg] without assignment expression '='"},
    # Negative example
    {"role": "user", "content": "Code idiom [char *message; if (condition) { message = (char *)malloc(len); }] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "no. Satisfies [Variable Type][Variable Name], and only declares pointer [message] without initialization"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [int value = 0;] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "else. There is no declaration of pointer variable, resource descriptor variable, or BOOL variable in the code idiom"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [_value.SetStr(buf)] Does it comply with the programming standard: [Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared"},
    {"role": "assistant", "content": "else. There is no declaration of pointer variable, resource descriptor variable, or BOOL variable in the code idiom"}
]

standard_prompt_2 = [
    # 1. Programming standard definition
    {"role": "system", "content": (
        "You are an expert in C++ programming and follow excellent programming standards. One variable-related programming standard is: Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released. Below is a code idiom, please determine whether it reflects the above programming standard.\n"
        "If the given code idiom reflects this programming standard well, return <yes>; if it violates this programming standard, return <no>; if the code idiom is irrelevant to this programming standard, return <else>\n"
        "\n"
        "Here are some example dialogues:"
    )},
    # 2. Positive and negative examples for this programming standard
    # Positive example
    {"role": "user", "content": "Code idiom [closesocket(s); s = INVALID_SOCKET;] Does it comply with the programming standard: [Variable Rule] Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released"},
    {"role": "assistant", "content": "yes. Variable [s] is a resource descriptor variable, and after releasing the resource with [closesocket(s);], it is immediately assigned a new value [INVALID_SOCKET]"},
    # Positive example
    {"role": "user", "content": "Code idiom [free(msg); msg = (unsigned char *)malloc(...);] Does it comply with the programming standard: [Variable Rule] Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released"},
    {"role": "assistant", "content": "yes. Variable [msg] is a resource descriptor variable, and after releasing the resource with [free(msg);], it is immediately assigned a new value [(unsigned char *)malloc(...)]"},
    # Negative example
    {"role": "user", "content": "Code idiom [closesocket(s);] Does it comply with the programming standard: [Variable Rule] Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released"},
    {"role": "assistant", "content": "no. Variable [s] is a resource descriptor variable, and after releasing the resource with [closesocket(s);], it is not immediately assigned a new value"},
    # Negative example
    {"role": "user", "content": "Code idiom [free(msg);] Does it comply with the programming standard: [Variable Rule] Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released"},
    {"role": "assistant", "content": "no. Variable [msg] is a resource descriptor variable, and after releasing the resource with [free(msg);], it is not immediately assigned a new value"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [int value = 0;] Does it comply with the programming standard: [Variable Rule] Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released"},
    {"role": "assistant", "content": "else. There is no resource release of resource handle or descriptor variable in the code idiom"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [_value.SetStr(buf)] Does it comply with the programming standard: [Variable Rule] Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released"},
    {"role": "assistant", "content": "else. There is no resource release of resource handle or descriptor variable in the code idiom"}
]

standard_prompt_3 = [
    # 1. Programming standard definition
    {"role": "system", "content": (
        "You are an expert in C++ programming and follow excellent programming standards. One function-related programming standard is: When an array is used as a function parameter, its length must also be passed as a parameter. Below is a code idiom, please determine whether it reflects the above programming standard.\n"
        "If the given code idiom reflects this programming standard well, return <yes>; if it violates this programming standard, return <no>; if the code idiom is irrelevant to this programming standard, return <else>\n"
        "\n"
        "Here are some example dialogues:"
    )},
    # 2. Positive and negative examples for this programming standard
    # Positive example
    {"role": "user", "content": "Code idiom [int ParseMsg(BYTE *msg, size_t msgLen) { ASSERT(msg != NULL); ASSERT(msgLen != 0); return 0;}] Does it comply with the programming standard: [Function Rule] When an array is used as a function parameter, its length must also be passed as a parameter"},
    {"role": "assistant", "content": "yes. When defining the function [ParseMsg], [BYTE *msg] is used as the array parameter, and [size_t msgLen] is used as the array length parameter"},
    # Positive example
    {"role": "user", "content": "Code idiom [ParseMsg(msg, len);] Does it comply with the programming standard: [Function Rule] When an array is used as a function parameter, its length must also be passed as a parameter"},
    {"role": "assistant", "content": "yes. When calling the function [ParseMsg], [msg] is used as the array argument, and [len] is used as the array length argument"},
    # Negative example
    {"role": "user", "content": "Code idiom [int ParseMsg(BYTE *msg) { ASSERT(msg != NULL); return 0;}] Does it comply with the programming standard: [Function Rule] When an array is used as a function parameter, its length must also be passed as a parameter"},
    {"role": "assistant", "content": "no. When defining the function [ParseMsg], [BYTE *msg] is used as the array parameter, but the array length parameter is not defined"},
    # Negative example
    {"role": "user", "content": "Code idiom [ParseMsg(msg);] Does it comply with the programming standard: [Function Rule] When an array is used as a function parameter, its length must also be passed as a parameter"},
    {"role": "assistant", "content": "no. When calling the function [ParseMsg], [msg] is used as the array argument, but the array length argument is not passed"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [int calculate(int x) { return x * x;}] Does it comply with the programming standard: [Function Rule] When an array is used as a function parameter, its length must also be passed as a parameter"},
    {"role": "assistant", "content": "else. There is no array-related parameter in the function definition"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [_value.calculate(x)] Does it comply with the programming standard: [Function Rule] When an array is used as a function parameter, its length must also be passed as a parameter"},
    {"role": "assistant", "content": "else. There is no array-related argument in the function call"}
]

standard_prompt_4 = [
    # 1. Programming standard definition
    {"role": "system", "content": (
        "You are an expert in C++ programming and follow excellent programming standards. One memory-related programming standard is: After memory allocation, you must check whether it was successful. Below is a code idiom, please determine whether it reflects the above programming standard.\n"
        "If the given code idiom reflects this programming standard well, return <yes>; if it violates this programming standard, return <no>; if the code idiom is irrelevant to this programming standard, return <else>\n"
        "\n"
        "Here are some example dialogues:"
    )},
    # 2. Positive and negative examples for this programming standard
    # Positive example
    {"role": "user", "content": "Code idiom [char *msg = (char *)malloc(size); if (msg != NULL) { ... }] Does it comply with the programming standard: [Memory Rule] After memory allocation, you must check whether it was successful"},
    {"role": "assistant", "content": "yes. After allocating memory to [msg], it immediately checks whether it was successful [if (msg != NULL)]"},
    # Negative example
    {"role": "user", "content": "Code idiom [char *msg = (char *)malloc(size);] Does it comply with the programming standard: [Memory Rule] After memory allocation, you must check whether it was successful"},
    {"role": "assistant", "content": "no. After allocating memory to [msg], it does not immediately check whether it was successful"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [int value = 0;] Does it comply with the programming standard: [Memory Rule] After memory allocation, you must check whether it was successful"},
    {"role": "assistant", "content": "else. There is no memory allocation operation in the code idiom"}
]

standard_prompt_5 = [
    # 1. Programming standard definition
    {"role": "system", "content": (
        "You are an expert in C++ programming and follow excellent programming standards. One memory-related programming standard is: After memory is released, a new value must be assigned immediately. Below is a code idiom, please determine whether it reflects the above programming standard.\n"
        "If the given code idiom reflects this programming standard well, return <yes>; if it violates this programming standard, return <no>; if the code idiom is irrelevant to this programming standard, return <else>\n"
        "\n"
        "Here are some example dialogues:"
    )},
    # 2. Positive and negative examples for this programming standard
    # Positive example
    {"role": "user", "content": "Code idiom [if (message != NULL) { free(message); message = NULL;}] Does it comply with the programming standard: [Memory Rule] After memory is released, a new value must be assigned immediately"},
    {"role": "assistant", "content": "yes. After releasing memory for [message], it is immediately assigned a new value [NULL]"},
    # Negative example
    {"role": "user", "content": "Code idiom [if (message != NULL) { free(message); }] Does it comply with the programming standard: [Memory Rule] After memory is released, a new value must be assigned immediately"},
    {"role": "assistant", "content": "no. After releasing memory for [message], it is not immediately assigned a new value"},
    # Irrelevant example
    {"role": "user", "content": "Code idiom [int value = 0;] Does it comply with the programming standard: [Memory Rule] After memory is released, a new value must be assigned immediately"},
    {"role": "assistant", "content": "else. There is no memory release operation in the code idiom"}
]

standard_prompts = [standard_prompt_1, standard_prompt_2, standard_prompt_3, standard_prompt_4, standard_prompt_5]

standard_1 = '[Variable Rule] Pointer variables, variables representing resource descriptors, and BOOL variables must be initialized when declared' # Negative examples are filtered out during idiom judgment
standard_2 = '[Variable Rule] Variables pointing to resource handles or descriptors must be assigned a new value immediately after the resource is released'
standard_3 = '[Function Rule] When an array is used as a function parameter, its length must also be passed as a parameter'
standard_4 = '[Memory Rule] After memory allocation, you must check whether it was successful'
standard_5 = '[Memory Rule] After memory is released, a new value must be assigned immediately'

standards = [standard_1, standard_2, standard_3, standard_4, standard_5]