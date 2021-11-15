//gcc -fPIC -shared -o libDemo3.so cchess_env.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_SIZE 1024
#define STATES_SIZE 150
#define MAX_SIZE_LINE 10
#define COL_LENGTH 9
#define ROW_LENGTH 10
#define POS_LABEL_LENGTH 3
#define TRUE 1
#define FALSE 0
#define MAX_VALID_ACTION 120
#define MAX_VALID_ACTION_STR 1200

void expand_num(char * single_line);
void compress_num(char * single_line);
void split_string(char * in_string, char res[][COL_LENGTH+1]);
void sim_do_action(char * in_action, char * in_state, char * res);
void create_position_labels(char pos_label[][POS_LABEL_LENGTH]);
int is_check_catch(char * in_state, char * next_player);
int game_end(char * in_state, char * current_player, char * winner);
void get_legal_target_pos(char * in_state, char * current_player, char res_pos[][POS_LABEL_LENGTH], int * legal_action_num);
void get_legal_action(char * in_state, char * current_player, char * res);
int check_bounds(int to_y, int to_x);
int validate_move(char c, int upper);
void append_action(int x, int y, int to_x, int to_y, char * res);
void rook_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res);
void knight_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res);
void bishop_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res);
void assistant_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res);
void king_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res);
void cannon_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res);
void pawn_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res);
void cannon_legal_action_single_direction(
    char board_positions[][10], int current_x, int current_y, int to_x, int to_y,
    char * current_player, char * res, int * hits_p, int * done_p);
void get_king_pos(char * in_state, char * w_k_pos, char * b_k_pos);


// global
char board_pos_name[ROW_LENGTH * COL_LENGTH][POS_LABEL_LENGTH] = {0};
int init_board_pos_name = FALSE;


void expand_num(char * single_line){
    // "p3p3p" -> "p111p111p111p"
    char *res = (char *) malloc(MAX_SIZE_LINE);
    int length = strlen(single_line);
    int index_ori;
    int index_res = 0;
    for(index_ori = 0; index_ori < length;index_ori++){
        if(isdigit(single_line[index_ori])){
            int i;
            int num = single_line[index_ori] - '0';
            for(i=0;i<num;i++){
                res[index_res]='1';
                index_res++;
            }
        }
        else{
            res[index_res]=single_line[index_ori];
            index_res++;
        }
        res[index_res] = '\0';

    }

    memset(single_line, '\0', sizeof(single_line));
    strcpy(single_line, res);
    free(res);
}

void compress_num(char * single_line){
    // "p111p111p111p" -> "p3p3p"
    char *res = (char *)malloc(MAX_SIZE_LINE);
    int index;
    int index_res = 0;
    int num_length = 0;
    for(index = 0;index < strlen(single_line); index++){
        if(isalpha(single_line[index])){
            if(num_length != 0){
                res[index_res] = '0' + num_length;
                index_res++;
                num_length = 0;
            }
            res[index_res] = single_line[index];
            index_res++;
        }
        else{
            num_length += 1;
        }
    }
    if(num_length != 0){
        res[index_res] = '0' + num_length;
        num_length = 0;
        index_res++;
    }
    res[index_res] = '\0';

    memset(single_line, '\0', sizeof(single_line));
    strcpy(single_line, res);
    free(res);
}

void split_string(char * in_string, char res[][COL_LENGTH+1]){
    // "RNBAKABNR/9/1C5C1/" -> {"RNBAKABNR", "9", "1C5C1"}
    char * in_state_copy = (char *) malloc(STATES_SIZE);
    strcpy(in_state_copy, in_string);

    const char s[2] = "/";
    char *token;
    int index = 0;
    token = strtok(in_state_copy, s);
    strcpy(res[index], token);

    while(token != NULL) {
        token = strtok(NULL, s);
        if(token != NULL){
            index++;
            strcpy(res[index], token);
        }
    }

    free(in_state_copy);
}

void sim_do_action(char * in_action, char * in_state, char * res)
{
    int src_x = in_action[0] - 'a';
    int src_y = in_action[1] - '0';
    int dst_x = in_action[2] - 'a';
    int dst_y = in_action[3] - '0';

    char board[ROW_LENGTH][COL_LENGTH+1] = {0};
    split_string(in_state, board);

    if(dst_y != src_y){
        char *board_src_y = (char *) malloc(COL_LENGTH+1);
        strcpy(board_src_y, board[src_y]);
        expand_num(board_src_y);
        char *board_dst_y = (char *) malloc(COL_LENGTH+1);
        strcpy(board_dst_y, board[dst_y]);
        expand_num(board_dst_y);
        board_dst_y[dst_x] = board_src_y[src_x];
        board_src_y[src_x] = '1';

        compress_num(board_src_y);
        compress_num(board_dst_y);

        memset(board[src_y], '\0', sizeof(board[src_y]));
        strcpy(board[src_y], board_src_y);
        memset(board[dst_y], '\0', sizeof(board[dst_y]));
        strcpy(board[dst_y], board_dst_y);

        free(board_src_y);
        free(board_dst_y);
    }
    else{
        char *board_line = (char *) malloc(COL_LENGTH+1);
        strcpy(board_line, board[src_y]);
        expand_num(board_line);
        board_line[dst_x] = board_line[src_x];
        board_line[src_x] = '1';

        compress_num(board_line);

        memset(board[src_y], '\0', sizeof(board[src_y]));
        strcpy(board[src_y], board_line);

        free(board_line);
    }

    memset(res, '\0', sizeof(res));
    strcpy(res, board[0]);
    int i;
    const char ch[2] = "/";
    for(i = 1 ; i < ROW_LENGTH ; i++){
        strcat(res, ch);
        strcat(res, board[i]);
    }
}

void create_position_labels(char pos_label[][POS_LABEL_LENGTH]){
    // output
    // "a0" "b0" "c0" "d0" "e0" "f0" "g0" "h0" "i0" 下同
    // a1 b1 c1 d1 e1 f1 g1 h1 i1
    // a2 b2 c2 d2 e2 f2 g2 h2 i2
    // a3 b3 c3 d3 e3 f3 g3 h3 i3
    // a4 b4 c4 d4 e4 f4 g4 h4 i4
    // a5 b5 c5 d5 e5 f5 g5 h5 i5
    // a6 b6 c6 d6 e6 f6 g6 h6 i6
    // a7 b7 c7 d7 e7 f7 g7 h7 i7
    // a8 b8 c8 d8 e8 f8 g8 h8 i8
    // a9 b9 c9 d9 e9 f9 g9 h9 i9
    char letters[COL_LENGTH][2] = {"a", "b", "c", "d", "e", "f", "g", "h", "i"};
    char numbers[ROW_LENGTH][2] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    int col, row;
    for(row = 0 ; row < ROW_LENGTH ; row++){
        for(col = 0 ; col < COL_LENGTH ; col++){
            char current_pos_label[POS_LABEL_LENGTH] = {0};
            strcpy(current_pos_label, letters[col]);
            strcat(current_pos_label, numbers[row]);
            strcpy(pos_label[row * COL_LENGTH + col], current_pos_label);
        }
    }
}

int is_check_catch(char * in_state, char * next_player){
    char * w_k_pos = (char *)malloc(POS_LABEL_LENGTH);
    char * b_k_pos = (char *)malloc(POS_LABEL_LENGTH);
    char * target_k_pos = (char *)malloc(POS_LABEL_LENGTH);
    get_king_pos(in_state, w_k_pos, b_k_pos);
    if(strcmp(next_player, "w") == 0){
        strcpy(target_k_pos, b_k_pos);
    }
    else{
        strcpy(target_k_pos, w_k_pos);
    }

    char legal_target_pos[MAX_VALID_ACTION][POS_LABEL_LENGTH] = {0};
    int init_legal_num = 0;
    int * legal_action_num = &init_legal_num;
    get_legal_target_pos(in_state, next_player, legal_target_pos, legal_action_num);
    int i;
    for(i = 0 ; i <= *legal_action_num ; i++){
        if(strcmp(target_k_pos, legal_target_pos[i]) == 0){
            free(w_k_pos);
            free(b_k_pos);
            free(target_k_pos);
            return TRUE;
        }
    }

    free(w_k_pos);
    free(b_k_pos);
    free(target_k_pos);
    return FALSE;
}

int game_end(char * in_state, char * current_player, char * winner){

    int in_state_length = strlen(in_state);
    int i;
    int b_k_exist = FALSE;
    int w_k_exist = FALSE;
    for(i = 0 ; i < in_state_length ; i++){
        if(in_state[i] == 'K'){
            w_k_exist = TRUE;
        }
        else if(in_state[i] == 'k'){
            b_k_exist = TRUE;
        }
    }

    if(b_k_exist == FALSE){
        strcpy(winner, "w");
        return TRUE;
    }
    if(w_k_exist == FALSE){
        strcpy(winner, "b");
        return TRUE;
    }

    char * w_k_pos = (char *)malloc(POS_LABEL_LENGTH);
    char * b_k_pos = (char *)malloc(POS_LABEL_LENGTH);
    char * target_k_pos = (char *)malloc(POS_LABEL_LENGTH);
    get_king_pos(in_state, w_k_pos, b_k_pos);
    if(strcmp(current_player, "w") == 0){
        strcpy(target_k_pos, b_k_pos);
    }
    else{
        strcpy(target_k_pos, w_k_pos);
    }

    char legal_target_pos[MAX_VALID_ACTION][POS_LABEL_LENGTH] = {0};
    int init_legal_num = 0;
    int * legal_action_num = &init_legal_num;
    get_legal_target_pos(in_state, current_player, legal_target_pos, legal_action_num);
    for(i = 0 ; i <= *legal_action_num ; i++){
        if(strcmp(target_k_pos, legal_target_pos[i]) == 0){
            strcpy(winner, current_player);
            free(w_k_pos);
            free(b_k_pos);
            free(target_k_pos);
            return TRUE;
        }
    }

    free(w_k_pos);
    free(b_k_pos);
    free(target_k_pos);

    strcpy(winner, "NONE");
    return FALSE;
}

void get_legal_target_pos(char * in_state, char * current_player, char res_pos[][POS_LABEL_LENGTH], int * legal_action_num){
    char * res = (char *)malloc(MAX_VALID_ACTION_STR);
    get_legal_action(in_state, current_player, res);

    const char s[2] = "/";
    char *token;

    int index = 0;
    token = strtok(res, s);
    if(token != NULL){
        res_pos[index][0] = token[2];
        res_pos[index][1] = token[3];
        res_pos[index][2] = '\0';
    }

    while(token != NULL) {
        token = strtok(NULL, s);
        if(token != NULL){
            index++;
            res_pos[index][0] = token[2];
            res_pos[index][1] = token[3];
            res_pos[index][2] = '\0';
        }
    }

    *legal_action_num = index;

    free(res);
}

void get_legal_action(char * in_state, char * current_player, char * res){
    if(init_board_pos_name == FALSE){
        create_position_labels(board_pos_name);
        init_board_pos_name = TRUE;
    }

    int w_k_x = -1;
    int b_k_x = -1;
    int w_k_y = -1;
    int b_k_y = -1;

    int face_to_face = FALSE;

    // get board_positions
    char board_positions[10][10] = {0};
    split_string(in_state, board_positions);
    int i;
    for(i = 0 ; i < 10 ; i++){
        expand_num(board_positions[i]);
    }

//    memset(res, '\0', sizeof(res));
    strcpy(res, "/");
    int y, x;
    for(y = 0 ; y < 10 ; y++){
        for(x = 0 ; x < 9 ; x++){
            if(isalpha(board_positions[y][x])){
                // 车
                if(
                    (board_positions[y][x] == 'r' && strcmp(current_player, "b") == 0) ||
                    (board_positions[y][x] == 'R' && strcmp(current_player, "w") == 0)
                ){
                    rook_legal_action(board_positions, x, y, current_player, res);
                }
                // 马
                else if(
                    ((board_positions[y][x] == 'n' || board_positions[y][x] == 'h') && strcmp(current_player, "b") == 0) ||
                    ((board_positions[y][x] == 'N' || board_positions[y][x] == 'H') && strcmp(current_player, "w") == 0)
                ){
                    knight_legal_action(board_positions, x, y, current_player, res);
                }
                // 象
                else if(
                    ((board_positions[y][x] == 'b' || board_positions[y][x] == 'e') && strcmp(current_player, "b") == 0) ||
                    ((board_positions[y][x] == 'B' || board_positions[y][x] == 'E') && strcmp(current_player, "w") == 0)
                ){
                    bishop_legal_action(board_positions, x, y, current_player, res);
                }
                // 士
                else if(
                    (board_positions[y][x] == 'a' && strcmp(current_player, "b") == 0) ||
                    (board_positions[y][x] == 'A' && strcmp(current_player, "w") == 0)
                ){
                    assistant_legal_action(board_positions, x, y, current_player, res);
                }
                // 将帅
                if(board_positions[y][x] == 'k' || board_positions[y][x] == 'K'){
                    if(board_positions[y][x] == 'k'){
                        b_k_x = x;
                        b_k_y = y;
                    }
                    else{
                        w_k_x = x;
                        w_k_y = y;
                    }

                    king_legal_action(board_positions, x, y, current_player, res);
                }
                // 炮
                else if(board_positions[y][x] == 'c' && strcmp(current_player, "b") == 0 ||
                        board_positions[y][x] == 'C' && strcmp(current_player, "w") == 0 ){
                    cannon_legal_action(board_positions, x, y, current_player, res);
                }
                // 兵
                else if(
                    (board_positions[y][x] == 'p' && strcmp(current_player, "b") == 0) ||
                    (board_positions[y][x] == 'P' && strcmp(current_player, "w") == 0)
                ){
                    pawn_legal_action(board_positions, x, y, current_player, res);
                }


            }

        }
    }

    if(w_k_x != -1 && b_k_x != -1 && w_k_x == b_k_x){
        face_to_face = TRUE;
        int i;
        for(i = w_k_y + 1 ; i < b_k_y ; i++){
            if(isalpha(board_positions[i][w_k_x])){
                face_to_face = FALSE;
                break;
            }
        }
    }
    if(face_to_face == TRUE){
        if(strcmp(current_player, "b") == 0){
            append_action(b_k_x, b_k_y, w_k_x, w_k_y, res);
        }
        else{
            append_action(w_k_x, w_k_y, b_k_x, b_k_y, res);
        }
    }
}

int check_bounds(int to_y, int to_x){
    if(to_y < 0 || to_x < 0){
        return FALSE;
    }
    else if(to_y >= ROW_LENGTH || to_x >= COL_LENGTH){
        return FALSE;
    }
    else{
        return TRUE;
    }
}

int validate_move(char c, int upper){
    if(isalpha(c)){
        if(upper == TRUE){
            if(islower(c)){
                return TRUE;
            }
            else{
                return FALSE;
            }
        }
        else{
            if(isupper(c)){
                return TRUE;
            }
            else{
                return FALSE;
            }
        }
    }
    else{
        return TRUE;
    }
}

void append_action(int x, int y, int to_x, int to_y, char * res){
    char move[5];
    strcpy(move, board_pos_name[y * COL_LENGTH + x]);
    strcat(move, board_pos_name[to_y * COL_LENGTH + to_x]);
    strcat(res, move);
    strcat(res, "/");
}

void rook_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res){
    int to_y = current_y;
    int to_x;
    for(to_x = current_x - 1 ; to_x >= 0 ; to_x--){
        if(isalpha(board_positions[to_y][to_x])){
            if((strcmp(current_player, "b") == 0 && isupper(board_positions[to_y][to_x])) ||
               (strcmp(current_player, "w") == 0 && islower(board_positions[to_y][to_x]))){
                append_action(current_x, current_y, to_x, to_y, res);
            }
            break;
        }

        append_action(current_x, current_y, to_x, to_y, res);
    }

    for(to_x = current_x + 1 ; to_x < COL_LENGTH ; to_x++){
        if(isalpha(board_positions[to_y][to_x])){
            if((strcmp(current_player, "b") == 0 && isupper(board_positions[to_y][to_x])) ||
               (strcmp(current_player, "w") == 0 && islower(board_positions[to_y][to_x]))){
                append_action(current_x, current_y, to_x, to_y, res);
            }

            break;
        }

        append_action(current_x, current_y, to_x, to_y, res);
    }

    to_x = current_x;
    for(to_y = current_y - 1 ; to_y >= 0 ; to_y--){
        if(isalpha(board_positions[to_y][to_x])){
            if((strcmp(current_player, "b") == 0 && isupper(board_positions[to_y][to_x])) ||
               (strcmp(current_player, "w") == 0 && islower(board_positions[to_y][to_x]))){
                append_action(current_x, current_y, to_x, to_y, res);
            }

            break;
        }

        append_action(current_x, current_y, to_x, to_y, res);
    }

    for(to_y = current_y + 1 ; to_y < ROW_LENGTH ; to_y++){
        if(isalpha(board_positions[to_y][to_x])){
            if((strcmp(current_player, "b") == 0 && isupper(board_positions[to_y][to_x])) ||
               (strcmp(current_player, "w") == 0 && islower(board_positions[to_y][to_x]))){
                append_action(current_x, current_y, to_x, to_y, res);
            }

            break;
        }

        append_action(current_x, current_y, to_x, to_y, res);
    }
}

void knight_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res){
    int i, j;
    for(i = -1 ; i < 3 ; i += 2){
        for(j = -1 ; j < 3 ; j += 2){
            int to_y = current_y + 2 * i;
            int to_x = current_x + 1 * j;
            if(check_bounds(to_y, to_x)){
                if(
                    (strcmp(current_player, "b") == 0 && validate_move(board_positions[to_y][to_x], FALSE) && isalpha(board_positions[current_y + i][current_x]) == FALSE) ||
                    (strcmp(current_player, "w") == 0 && validate_move(board_positions[to_y][to_x], TRUE) && isalpha(board_positions[current_y + i][current_x]) == FALSE)
                ){
                    append_action(current_x, current_y, to_x, to_y, res);
                }
            }

            to_y = current_y + 1 * i;
            to_x = current_x + 2 * j;
            if(check_bounds(to_y, to_x)){
                if(
                    (strcmp(current_player, "b") == 0 && validate_move(board_positions[to_y][to_x], FALSE) && isalpha(board_positions[current_y][current_x + j]) == FALSE) ||
                    (strcmp(current_player, "w") == 0 && validate_move(board_positions[to_y][to_x], TRUE) && isalpha(board_positions[current_y][current_x + j]) == FALSE)
                ){
                    append_action(current_x, current_y, to_x, to_y, res);
                }
            }
        }
    }
}

void bishop_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res){
    int i;
    for(i = -2 ; i < 3 ; i += 4){
        int to_y = current_y + i;
        int to_x = current_x + i;
        if(check_bounds(to_y, to_x)){
            if(
                (strcmp(current_player, "b") == 0 && validate_move(board_positions[to_y][to_x], FALSE) && to_y >= 5 && isalpha(board_positions[current_y + i / 2][current_x + i / 2]) == FALSE) ||
                (strcmp(current_player, "w") == 0 && validate_move(board_positions[to_y][to_x], TRUE) && to_y <= 4 && isalpha(board_positions[current_y + i / 2][current_x + i / 2]) == FALSE)
            ){
                append_action(current_x, current_y, to_x, to_y, res);
            }
        }

        to_y = current_y + i;
        to_x = current_x - i;
        if(check_bounds(to_y, to_x)){
            if(
                (strcmp(current_player, "b") == 0 && validate_move(board_positions[to_y][to_x], FALSE) && to_y >= 5 && isalpha(board_positions[current_y + i / 2][current_x - i / 2]) == FALSE) ||
                (strcmp(current_player, "w") == 0 && validate_move(board_positions[to_y][to_x], TRUE) && to_y <= 4 && isalpha(board_positions[current_y + i / 2][current_x - i / 2]) == FALSE)
            ){
                append_action(current_x, current_y, to_x, to_y, res);
            }
        }
    }
}

void assistant_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res){
    int i;
    for(i = -1; i < 3 ; i += 2){
        int to_y = current_y + i;
        int to_x = current_x + i;
        if(check_bounds(to_y, to_x)){
            if(
                (strcmp(current_player, "b") == 0 && validate_move(board_positions[to_y][to_x], FALSE) && to_y >= 7 && 3 <= to_x && to_x <= 5) ||
                (strcmp(current_player, "w") == 0 && validate_move(board_positions[to_y][to_x], TRUE) && to_y <= 2 && 3 <= to_x && to_x <= 5)
            ){
                append_action(current_x, current_y, to_x, to_y, res);
            }
        }

        to_y = current_y + i;
        to_x = current_x - i;
        if(check_bounds(to_y, to_x)){
            if(
                (strcmp(current_player, "b") == 0 && validate_move(board_positions[to_y][to_x], FALSE) && to_y >= 7 && 3 <= to_x && to_x <= 5) ||
                (strcmp(current_player, "w") == 0 && validate_move(board_positions[to_y][to_x], TRUE) && to_y <= 2 && 3 <= to_x && to_x <= 5)
            ){
                append_action(current_x, current_y, to_x, to_y, res);
            }
        }
    }
}

void king_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res){
    int i;
    for(i = 0 ; i < 2 ; i++){
        int sign;
        for(sign = -1 ; sign < 2 ; sign += 2){
            int j = 1 - i;
            int to_y = current_y + i * sign;
            int to_x = current_x + j * sign;
            if(check_bounds(to_y, to_x)){
                if(
                    (strcmp(current_player, "b") == 0 && validate_move(board_positions[to_y][to_x], FALSE) && to_y >= 7 && 3 <= to_x && to_x <= 5) ||
                    (strcmp(current_player, "w") == 0 && validate_move(board_positions[to_y][to_x], TRUE) && to_y <= 2 && 3 <= to_x && to_x <= 5)
                ){
                    append_action(current_x, current_y, to_x, to_y, res);
                }
            }

        }
    }
}


void cannon_legal_action_single_direction(
    char board_positions[][10], int current_x, int current_y, int to_x, int to_y,
    char * current_player, char * res, int * hits_p, int * done_p){
    if(*hits_p == FALSE){
        if(isalpha(board_positions[to_y][to_x])){
            *hits_p = TRUE;
        }
        else{
            append_action(current_x, current_y, to_x, to_y, res);
        }
    }
    else{
        if(isalpha(board_positions[to_y][to_x])){
            if(
                (strcmp(current_player, "b") == 0 && isupper(board_positions[to_y][to_x])) ||
                (strcmp(current_player, "w") == 0 && islower(board_positions[to_y][to_x]))
            ){
                append_action(current_x, current_y, to_x, to_y, res);
            }

            *done_p = TRUE;
        }
    }
}

void cannon_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res){
    int hits = FALSE;
    int done_search = FALSE;
    int *hits_p = &hits;
    int *done_p = &done_search;

    int to_y = current_y;
    int to_x;
    for(to_x = current_x - 1 ; to_x > -1 ; to_x--){
        cannon_legal_action_single_direction(
            board_positions, current_x, current_y, to_x, to_y, current_player, res, hits_p, done_p
        );
        if(*done_p == TRUE){
            break;
        }
    }

    *hits_p = FALSE;
    *done_p = FALSE;
    for(to_x = current_x + 1 ; to_x < COL_LENGTH ; to_x++){
        cannon_legal_action_single_direction(
            board_positions, current_x, current_y, to_x, to_y, current_player, res, hits_p, done_p
        );
        if(*done_p == TRUE){
            break;
        }
    }

    to_x = current_x;
    *hits_p = FALSE;
    *done_p = FALSE;
    for(to_y = current_y - 1 ; to_y > -1 ; to_y--){
        cannon_legal_action_single_direction(
            board_positions, current_x, current_y, to_x, to_y, current_player, res, hits_p, done_p
        );
        if(*done_p == TRUE){
            break;
        }
    }

    *hits_p = FALSE;
    *done_p = FALSE;
    for(to_y = current_y + 1 ; to_y < ROW_LENGTH ; to_y++){
        cannon_legal_action_single_direction(
            board_positions, current_x, current_y, to_x, to_y, current_player, res, hits_p, done_p
        );
        if(*done_p == TRUE){
            break;
        }
    }
}

void pawn_legal_action(char board_positions[][10], int current_x, int current_y, char * current_player, char * res){
    if(strcmp(current_player, "b") == 0){
        int to_y = current_y - 1;
        int to_x = current_x;
        if(check_bounds(to_y, to_x) && validate_move(board_positions[to_y][to_x], FALSE)){
            append_action(current_x, current_y, to_x, to_y, res);
        }

        if(current_y < 5){
            to_y = current_y;
            to_x = current_x + 1;
            if(check_bounds(to_y, to_x) && validate_move(board_positions[to_y][to_x], FALSE)){
                append_action(current_x, current_y, to_x, to_y, res);
            }

            to_x = current_x - 1;
            if(check_bounds(to_y, to_x) && validate_move(board_positions[to_y][to_x], FALSE)){
                append_action(current_x, current_y, to_x, to_y, res);
            }
        }
    }
    else if(strcmp(current_player, "w") == 0){
        int to_y = current_y + 1;
        int to_x = current_x;
        if(check_bounds(to_y, to_x) && validate_move(board_positions[to_y][to_x], TRUE)){
            append_action(current_x, current_y, to_x, to_y, res);
        }

        if(current_y > 4){
            to_y = current_y;
            to_x = current_x + 1;
            if(check_bounds(to_y, to_x) && validate_move(board_positions[to_y][to_x], TRUE)){
                append_action(current_x, current_y, to_x, to_y, res);
            }

            to_x = current_x - 1;
            if(check_bounds(to_y, to_x) && validate_move(board_positions[to_y][to_x], TRUE)){
                append_action(current_x, current_y, to_x, to_y, res);
            }
        }
    }
}

void get_king_pos(char * in_state, char * w_k_pos, char * b_k_pos){
    if(init_board_pos_name == FALSE){
        create_position_labels(board_pos_name);
        init_board_pos_name = TRUE;
    }

    int i;
    int in_state_len = strlen(in_state);
    int x = 0;
    int y = 0;

    int w_k_exist = FALSE;
    int b_k_exist = FALSE;

    for(i = 0 ; i < in_state_len ; i++){
        if(in_state[i] == 'K'){
            strcpy(w_k_pos, board_pos_name[y * COL_LENGTH + x]);
            w_k_exist = TRUE;
        }
        else if(in_state[i] == 'k'){
            strcpy(b_k_pos, board_pos_name[y * COL_LENGTH + x]);
            b_k_exist = TRUE;
        }

        if(isdigit(in_state[i])){
            x += in_state[i] - '0';
        }
        else if(isalpha(in_state[i])){
            x += 1;
        }
        else if(in_state[i] == '/'){
            y += 1;
            x = 0;
        }
    }

    if(w_k_exist == FALSE){
        strcpy(w_k_pos, "NONE");
    }
    if(b_k_exist == FALSE){
        strcpy(b_k_pos, "NONE");
    }
}

//释放函数
void freePoint(void *pt) {
    if (pt != NULL) {
        free(pt);
        pt = NULL;
    }
}


int main()
{
//    create_position_labels(board_pos_name);
//
////    int i;
////    for(i = 0 ; i < ROW_LENGTH * COL_LENGTH ; i++){
////        printf("%s ", board_pos_name[i]);
////    }
//
////    char board_positions[10][10] = {0};
////    split_string("RNBA1ABNR/4K4/1C5C1/P1PP2P1P/9/9/p1pp2p1p/1c5c1/4k4/rnba1abnr", board_positions);
////    int i;
////    for(i = 0 ; i < 10 ; i++){
////        expand_num(board_positions[i]);
////    }
//
////    char * res = (char *)malloc(10000 * sizeof(char));
////    char current_player[2] = "w";
//
////    int x, y;
////    for(y = 0 ; y < 10 ; y++){
////        for(x = 0 ; x < 9 ; x++){
////            if(board_positions[y][x] == 'k' || board_positions[y][x] == 'K'){
////                king_legal_action(board_positions, x, y, current_player, res);
////            }
////        }
////    }
////
////    get_legal_action("RNBA1ABNR/4K4/1C5C1/P1PP2P1P/9/9/p1pp2p1p/1c5c1/4k4/rnba1abnr", "w", res);
////
////    printf("%s", res);
////    free(res);
////
////    char * in_state = "RNBA1ABNR/4K4/1C5C1/P1PP2P1P/9/9/p1pp2p1p/1c5c1/4k4/rnba1abnr";
////    int i;
////    int length = strlen(in_state);
////    for(i = 0 ; i < length ; i++){
////        printf("%c ", in_state[i]);
////    }
////    printf("%d", isalpha('/'));
//
////    char legal_target_pos[ROW_LENGTH*COL_LENGTH][POS_LABEL_LENGTH] = {0};
////    int * legal_action_num = 0;
////    get_legal_target_pos("RNBA1ABNR/4K4/1C5C1/P1PP2P1P/9/9/p1pp2p1p/1c5c1/4k4/rnba1abnr", "w", legal_target_pos, legal_action_num);
////    int i;
////    for(i = 0 ; i < *legal_action_num ; i++){
////        printf("%s", legal_target_pos[i]);
////    }
////
//    char * res_end = (char *)malloc(10 * sizeof(char));
//    int over = game_end("RNBA1ABNR/4K4/1C5C1/P1PP2P1P/9/9/p1pp2p1p/1c5c1/4k4/rnba1abnr", "w", res_end);
//    printf("%d\n", over);
//    free(res_end);
//
//    char * res_action = (char *)malloc(10000 * sizeof(char));
//    get_legal_action("RNBA1ABNR/4K4/1C5C1/P1PP2P1P/9/9/p1pp2p1p/1c5c1/4k4/rnba1abnr", "w", res_action);
//    printf("%s \n", res_action);
//    free(res_action);
//
//    char * res_sim = (char *)malloc(100 * sizeof(char));
//    sim_do_action("a0a1", "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1pp2p1p/1c5c1/9/rnbakabnr", res_sim);
//    printf("%s \n", res_sim);
//    free(res_sim);
//
//    int res = is_check_catch("RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1pp2p1p/1c5c1/9/rnbakabnr", "b");
//    printf("%d \n", res);
//
//
    return 0;
}