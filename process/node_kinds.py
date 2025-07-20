func_kind = {"CursorKind.FUNCTION_DECL", "CursorKind.CXX_METHOD", "CursorKind.CLASS_DECL",
             "CursorKind.CLASS_TEMPLATE", "CursorKind.FUNCTION_TEMPLATE"} # Function level (only for clustering, not for synthesis, needs to be refined)
block_kind = {"CursorKind.IF_STMT", "CursorKind.FOR_STMT", "CursorKind.WHILE_STMT", "CursorKind.DO_STMT", "CursorKind.CXX_FOR_RANGE_STMT",
             "CursorKind.CXX_TRY_STMT", "CursorKind.CXX_CATCH_STMT", "CursorKind.SWITCH_STMT"} # Additionally record the entire range, participate in synthesis
stmt_kind = {"CursorKind.DECL_STMT", "CursorKind.RETURN_STMT", "CursorKind.BINARY_OPERATOR", "CursorKind.CXX_THROW_EXPR",
             "CursorKind.FIELD_DECL", "CursorKind.CALL_EXPR", "CursorKind.LAMBDA_EXPR", # Statement level (participate in synthesis, cannot be repeated, keep the largest)
             "CursorKind.ASM_STMT", "CursorKind.LABEL_STMT", "CursorKind.GOTO_STMT", "CursorKind.INDIRECT_GOTO_STMT",
             "CursorKind.SEH_EXCEPT_STMT", "CursorKind.SEH_FINALLY_STMT", "CursorKind.SEH_LEAVE_STMT", "CursorKind.SEH_TRY_STMT"}

# print(f"func_kind: {func_kind}")
# print(f"range_kind: {block_kind}")
# print(f"stmt_kind: {stmt_kind}")