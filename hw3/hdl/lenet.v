`define WAIT 3'd0
`define CONV1 3'd1
`define CONV2 3'd2
`define CONV3 3'd3
`define FC1 3'd4
`define FC2 3'd5
`define FINISH 3'd6

module lenet (
    input wire clk,
    input wire rst_n,

    input wire compute_start,
    output reg compute_finish,

    // Quantization scale
    input wire [31:0] scale_CONV1,
    input wire [31:0] scale_CONV2,
    input wire [31:0] scale_CONV3,
    input wire [31:0] scale_FC1,
    input wire [31:0] scale_FC2,

    // Weight sram, dual port
    output reg [ 3:0] sram_weight_wea0,
    output reg [15:0] sram_weight_addr0,
    output reg [31:0] sram_weight_wdata0,
    input wire [31:0] sram_weight_rdata0,
    output reg [ 3:0] sram_weight_wea1,
    output reg [15:0] sram_weight_addr1,
    output reg [31:0] sram_weight_wdata1,
    input wire [31:0] sram_weight_rdata1,

    // Activation sram, dual port
    output reg [ 3:0] sram_act_wea0,
    output reg [15:0] sram_act_addr0,
    output reg [31:0] sram_act_wdata0,
    input wire [31:0] sram_act_rdata0,
    output reg [ 3:0] sram_act_wea1,
    output reg [15:0] sram_act_addr1,
    output reg [31:0] sram_act_wdata1,
    input wire [31:0] sram_act_rdata1
);
    // Add your design here 
    reg [2:0] state, next_state;
    wire load_done, layer_done, do_store;
    wire [8*8-1:0] in_act; // 8 8-bit
    wire [10*8-1:0] weight; // 10 8-bit
    wire [8*8-1:0] out_act; // 8 8-bit
    reg [9:0] scale;
    wire [7:0] wea;

    // act_addr - load, store
    wire [9:0] load_act_addr0, load_act_addr1, store_act_addr0, store_act_addr1;
    
    // input/output delay
    reg in_rst_n;
    reg in_compute_start;
    reg out_compute_finish;
    reg [9:0] in_scale_CONV1, in_scale_CONV2, in_scale_CONV3, in_scale_FC1;
    wire [13:0] out_sram_weight_addr0, out_sram_weight_addr1;
    reg [31:0] in_sram_weight_rdata0, in_sram_weight_rdata1;
    wire [3:0] out_sram_act_wea0, out_sram_act_wea1;
    reg [9:0] out_sram_act_addr0, out_sram_act_addr1;
    wire [31:0] out_sram_act_wdata0, out_sram_act_wdata1;
    reg [31:0] in_sram_act_rdata0, in_sram_act_rdata1;

    always @(posedge clk) begin
        // input delay
        in_rst_n <= rst_n;
        in_compute_start <= compute_start;
        in_scale_CONV1 <= scale_CONV1;
        in_scale_CONV2 <= scale_CONV2;
        in_scale_CONV3 <= scale_CONV3;
        in_scale_FC1 <= scale_FC1;
        in_sram_weight_rdata0 <= sram_weight_rdata0;
        in_sram_weight_rdata1 <= sram_weight_rdata1;
        in_sram_act_rdata0 <= sram_act_rdata0;
        in_sram_act_rdata1 <= sram_act_rdata1;

        // output delay
        compute_finish <= out_compute_finish;
        sram_weight_wea0 <= 1'b0; // no write
        sram_weight_addr0 <= out_sram_weight_addr0;
        sram_weight_wdata0 <= 1'b0; // no write
        sram_weight_wea1 <= 1'b0; // no write
        sram_weight_addr1 <= out_sram_weight_addr1;
        sram_weight_wdata1 <= 1'b0; // no write
        sram_act_wea0 <= out_sram_act_wea0;
        sram_act_addr0 <= out_sram_act_addr0;
        sram_act_wdata0 <= out_sram_act_wdata0;
        sram_act_wea1 <= out_sram_act_wea1;
        sram_act_addr1 <= out_sram_act_addr1;
        sram_act_wdata1 <= out_sram_act_wdata1;
    end  

    // delay signal
    reg [6*3-1:0] state_delay; // 6 3-bit delay
    reg [5:0] do_store_delay; // 6 delay
    reg [2:0] layer_done_delay; // 3 delay

    always @(posedge clk) begin
        if(!in_rst_n) begin
            state_delay <= 1'b0;
            do_store_delay <= 1'b0;
            layer_done_delay <= 1'b0;
        end
        else begin
            state_delay <= {state_delay[14:0], state[2:0]};
            do_store_delay <= {do_store_delay[4:0], do_store};
            layer_done_delay <= {layer_done_delay[1:0], layer_done};
        end
    end
      
    load_act lda(
        .clk(clk),
        .rst_n(in_rst_n),
        .state(state),
        .stop(do_store_delay[0]),
        .in_act(in_act),
        .load_done(load_done),
        .sram_act_addr0(load_act_addr0), .sram_act_addr1(load_act_addr1),
        .sram_act_rdata0(in_sram_act_rdata0), .sram_act_rdata1(in_sram_act_rdata1)     
    );

    load_weight ldw(
        .clk(clk),
        .rst_n(in_rst_n),
        .state(state),
        .stop(do_store_delay[0]),
        .weight(weight),
        .sram_weight_addr0(out_sram_weight_addr0), .sram_weight_addr1(out_sram_weight_addr1),
        .sram_weight_rdata0(in_sram_weight_rdata0), .sram_weight_rdata1(in_sram_weight_rdata1)    
    );

    layer ly(
        .clk(clk),
        .rst_n(in_rst_n),
        .state(state_delay[17:12]),
        .stop(do_store_delay[5:4]),
        .weight(weight),
        .in_act(in_act),
        .scale(scale),
        .out_act(out_act),
        .wea(wea),
        .do_store(do_store),
        .layer_done(layer_done)
    );

    store sd(
        .clk(clk),
        .rst_n(in_rst_n),
        .state(state_delay[17:15]),
        .do_store(do_store),
        .out_act(out_act),
        .wea(wea),
        .sram_act_wea0(out_sram_act_wea0), .sram_act_wea1(out_sram_act_wea1),
        .sram_act_addr0(store_act_addr0), .sram_act_addr1(store_act_addr1),
        .sram_act_wdata0(out_sram_act_wdata0), .sram_act_wdata1(out_sram_act_wdata1)    
    );

    always @(posedge clk) begin
        if(!in_rst_n) state <= `WAIT;       
        else state <= next_state;        
    end

    always @(*) begin    
        if(do_store_delay[0]) begin // store
            out_sram_act_addr0 = store_act_addr0;
            out_sram_act_addr1 = store_act_addr1;
        end
        else begin // load
            out_sram_act_addr0 = load_act_addr0;
            out_sram_act_addr1 = load_act_addr1;
        end
        
        // FSM
        case(state)
            `WAIT: begin
                next_state = in_compute_start ? `CONV1 : `WAIT;
                scale = 0;          
            end
            `CONV1: begin
                next_state = load_done? `CONV2: `CONV1;   
                scale = in_scale_CONV1;
            end      
            `CONV2: begin
                next_state = load_done? `CONV3: `CONV2;  
                scale = in_scale_CONV2;
            end  
            `CONV3: begin
                next_state = load_done? `FC1: `CONV3;   
                scale = in_scale_CONV3;
            end  
            `FC1: begin
                next_state = load_done? `FC2: `FC1;    
                scale = in_scale_FC1;
            end  
            `FC2: begin
                next_state = load_done? `FINISH: `FC2;     
                scale = in_scale_FC1; // no
            end  
            default: begin // `FINISH
                next_state = layer_done_delay[2]? `WAIT: `FINISH;
                scale = 0;
            end
        endcase

        case(state)
            `FINISH: out_compute_finish = layer_done_delay[2];
            default: out_compute_finish = 1'b0;    
        endcase
    end

endmodule

//

module load_act(
    input wire clk,
    input wire rst_n,
    input wire [2:0] state,
    input wire stop,
    output reg [8*8-1:0] in_act, // 8 8-bit
    output reg load_done,
    output reg [9:0] sram_act_addr0, sram_act_addr1,
    input wire [31:0] sram_act_rdata0, sram_act_rdata1
);

    reg [3:0] cnt_unit1, next_cnt_unit1;
    reg [6:0] cnt_unit2, next_cnt_unit2;
    reg [9:0] cnt_unit3, next_cnt_unit3;
    reg [13:0] cnt_unit4, next_cnt_unit4;
    reg unit1_bound, unit2_bound, unit3_bound, unit4_bound;
    reg [9:0] act_idx, next_act_idx;
    reg [8*8-1:0] next_in_act;
    reg [9:0] next_act_addr0, next_act_addr1;
    reg next_load_done;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            cnt_unit1 <= 1'b0;
            cnt_unit2 <= 1'b0;
            cnt_unit3 <= 1'b0;
            cnt_unit4 <= 1'b0;
            act_idx <= 1'b0;
            in_act <= 1'b0;
            sram_act_addr0 <= 16'b0;
            sram_act_addr1 <= 16'b1;
            load_done <= 1'b0;
        end
        else begin
            cnt_unit1 <= next_cnt_unit1;
            cnt_unit2 <= next_cnt_unit2;
            cnt_unit3 <= next_cnt_unit3;
            cnt_unit4 <= next_cnt_unit4;
            act_idx <= next_act_idx;
            in_act <= next_in_act;
            sram_act_addr0 <= next_act_addr0;
            sram_act_addr1 <= next_act_addr1;  
            load_done <= next_load_done; 
        end
    end

    always @(*) begin
        if(!stop) begin
            case(state)
                `CONV1: begin
                    unit1_bound = (cnt_unit1 == 4'd6-1);
                    unit2_bound = (cnt_unit2 == 7'd42-1); 
                    unit3_bound = (cnt_unit3 == 10'd588-1); 
                    unit4_bound = (cnt_unit4 == 14'd3528-1); 
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1);
                    next_cnt_unit4 = unit4_bound? 1'b0: (cnt_unit4 + 1'b1);
                    if(unit4_bound)
                        next_act_idx = 10'd256; // next layer
                    else begin
                        if(unit3_bound)
                            next_act_idx = 1'b0; // this layer
                        else begin
                            if(unit2_bound)
                                next_act_idx = act_idx + 10'd10; // next 2 row
                            else
                                next_act_idx = unit1_bound? (act_idx + 1'b1): act_idx; // next column
                        end        
                    end   
                    next_act_addr0 = act_idx + cnt_unit1*8; // same column, 6 continuous row
                    next_act_addr1 = act_idx + cnt_unit1*8 + 1'b1;              
                    next_in_act = {sram_act_rdata1, sram_act_rdata0};   
                    next_load_done = unit4_bound;
                end
                `CONV2: begin
                    unit1_bound = (cnt_unit1 == 4'd6-1);
                    unit2_bound = (cnt_unit2 == 7'd108-1);
                    unit3_bound = (cnt_unit3 == 10'd540-1);
                    unit4_bound = (cnt_unit4 == 14'd8640-1);
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1);
                    next_cnt_unit4 = unit4_bound? 1'b0: (cnt_unit4 + 1'b1);
                    if(unit4_bound)
                        next_act_idx = 10'd592; // next layer
                    else begin
                        if(unit3_bound) 
                            next_act_idx = 10'd256; // this layer
                        else begin
                            case(cnt_unit2)
                                7'd36-1, 7'd72-1: next_act_idx = act_idx - 10'd280 + 1'b1; // next column
                                7'd108-1: next_act_idx = act_idx - 10'd280 + 10'd6; // next 2 row
                                default: next_act_idx = unit1_bound? (act_idx + 10'd56): act_idx; // next channel
                            endcase  
                        end   
                    end
                    next_act_addr0 = act_idx + cnt_unit1*4; // same column, 6 continuous row
                    next_act_addr1 = act_idx + cnt_unit1*4 + 1'b1;  
                    next_in_act = {sram_act_rdata1, sram_act_rdata0}; 
                    next_load_done = unit4_bound;
                end
                `CONV3: begin
                    unit1_bound = (cnt_unit1 == 4'd5-1); // no
                    unit2_bound = (cnt_unit2 == 7'd5-1); // no
                    unit3_bound = (cnt_unit3 == 10'd50-1); 
                    unit4_bound = (cnt_unit4 == 14'd6000-1); 
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1); 
                    next_cnt_unit4 = unit4_bound? 1'b0: (cnt_unit4 + 1'b1);
                    next_act_idx = unit4_bound? 10'd692: act_idx; // next layer
                    next_act_addr0 = act_idx + cnt_unit3*2;
                    next_act_addr1 = act_idx + cnt_unit3*2 + 1'b1;  
                    next_in_act = {sram_act_rdata1, sram_act_rdata0}; 
                    next_load_done = unit4_bound;
                end
                `FC1: begin
                    unit1_bound = (cnt_unit1 == 4'd15-1); // no
                    unit2_bound = (cnt_unit2 == 7'd15-1); // no
                    unit3_bound = (cnt_unit3 == 10'd15-1); 
                    unit4_bound = (cnt_unit4 == 14'd1260-1);
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1); 
                    next_cnt_unit4 = unit4_bound? 1'b0: (cnt_unit4 + 1'b1);
                    next_act_idx =  unit4_bound? 10'd722: act_idx; // next layer
                    next_act_addr0 = act_idx + cnt_unit3*2;
                    next_act_addr1 = act_idx + cnt_unit3*2 + 1'b1;  
                    next_in_act = {sram_act_rdata1, sram_act_rdata0}; 
                    next_load_done = unit4_bound;
                end
                `FC2: begin
                    unit1_bound = (cnt_unit1 == 4'd11-1); // no
                    unit2_bound = (cnt_unit2 == 7'd11-1); // no
                    unit3_bound = (cnt_unit3 == 10'd11-1); 
                    unit4_bound = (cnt_unit4 == 14'd110-1);
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1); 
                    next_cnt_unit4 = unit4_bound? 1'b0: (cnt_unit4 + 1'b1);
                    next_act_idx = act_idx;
                    next_act_addr0 = act_idx + cnt_unit3*2;
                    next_act_addr1 = act_idx + cnt_unit3*2 + 1'b1;  
                    next_in_act = {sram_act_rdata1, sram_act_rdata0}; 
                    next_load_done = unit4_bound;
                end
                default: begin
                    unit1_bound = 1'b0;
                    unit2_bound = 1'b0;
                    unit3_bound = 1'b0;
                    unit4_bound = 1'b0;
                    next_cnt_unit1 = cnt_unit1;
                    next_cnt_unit2 = cnt_unit2;
                    next_cnt_unit3 = cnt_unit3;
                    next_cnt_unit4 = cnt_unit4;
                    next_act_idx = act_idx;
                    next_act_addr0 = sram_act_addr0;
                    next_act_addr1 = sram_act_addr1;
                    next_in_act = {sram_act_rdata1, sram_act_rdata0};
                    next_load_done = load_done;
                end
            endcase
        end
        else begin
            unit1_bound = 1'b0;
            unit2_bound = 1'b0;
            unit3_bound = 1'b0;
            unit4_bound = 1'b0;
            next_cnt_unit1 = cnt_unit1;
            next_cnt_unit2 = cnt_unit2;
            next_cnt_unit3 = cnt_unit3;
            next_cnt_unit4 = cnt_unit4;
            next_act_idx = act_idx;
            next_act_addr0 = sram_act_addr0;
            next_act_addr1 = sram_act_addr1;
            next_in_act = {sram_act_rdata1, sram_act_rdata0};
            next_load_done = load_done;
        end
    end

endmodule

//

module load_weight (
    input wire clk,
    input wire rst_n,
    input wire [2:0] state,
    input wire stop,
    output reg [10*8-1:0] weight, // 10 8-bit
    output reg [13:0] sram_weight_addr0, sram_weight_addr1,
    input wire [31:0] sram_weight_rdata0, sram_weight_rdata1
);

    reg [3:0] cnt_unit1, next_cnt_unit1;
    reg [5:0] cnt_unit2, next_cnt_unit2;
    reg [12:0] cnt_unit3, next_cnt_unit3;
    reg unit1_bound, unit2_bound, unit3_bound;
    reg [13:0] weight_idx, next_weight_idx;
    reg [10*8-1:0] next_weight;
    reg [13:0] next_weight_addr0, next_weight_addr1;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            cnt_unit1 <= 1'b0;
            cnt_unit2 <= 1'b0;
            cnt_unit3 <= 1'b0;
            weight_idx <= 1'b0;
            weight <= 1'b0;
            sram_weight_addr0 <= 16'b0;
            sram_weight_addr1 <= 16'b1;
        end
        else begin
            cnt_unit1 <= next_cnt_unit1;
            cnt_unit2 <= next_cnt_unit2;
            cnt_unit3 <= next_cnt_unit3;
            weight_idx <= next_weight_idx;
            weight <= next_weight;
            sram_weight_addr0 <= next_weight_addr0;
            sram_weight_addr1 <= next_weight_addr1;   
        end
    end

    always @(*) begin
        if(!stop) begin
            case(state)
                `CONV1: begin
                    unit1_bound = (cnt_unit1 == 4'd6-1);
                    unit2_bound = (cnt_unit2 == 6'd6-1); // no use
                    unit3_bound = (cnt_unit3 == 13'd588-1); 
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1);
                    next_weight_idx = unit3_bound? (weight_idx + 10'd10): weight_idx; // next filter
                    next_weight_addr0 = unit1_bound? sram_weight_addr0: (weight_idx + cnt_unit1*2); // 5 continuous row for 6 cycle
                    next_weight_addr1 = unit1_bound? sram_weight_addr1: (weight_idx + cnt_unit1*2 + 1'b1);                       
                end 
                `CONV2: begin
                    unit1_bound = (cnt_unit1 == 4'd6-1);
                    unit2_bound = (cnt_unit2 == 6'd36-1);
                    unit3_bound = (cnt_unit3 == 13'd540-1);
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1);
                    if(unit3_bound)
                        next_weight_idx = weight_idx + 10'd10; // next filter                   
                    else begin
                        if(unit2_bound)
                            next_weight_idx = weight_idx - 10'd50; // this filter
                        else
                            next_weight_idx = unit1_bound? (weight_idx + 10'd10): weight_idx; // next channel
                    end
                    next_weight_addr0 = unit1_bound? sram_weight_addr0: (weight_idx + cnt_unit1*2); // 5 continuous row for 6 cycle
                    next_weight_addr1 = unit1_bound? sram_weight_addr1: (weight_idx + cnt_unit1*2 + 1'b1);                           
                end
                `CONV3: begin
                    unit1_bound = (cnt_unit1 == 4'd6-1); // no
                    unit2_bound = (cnt_unit2 == 6'd6-1); // no
                    unit3_bound = (cnt_unit3 == 13'd6000-1);
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit2_bound? 1'b0: (cnt_unit2 + 1'b1);
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1);
                    next_weight_idx = unit3_bound? 16'd13020: weight_idx; // next layer
                    next_weight_addr0 = weight_idx + cnt_unit3*2;
                    next_weight_addr1 = weight_idx + cnt_unit3*2 + 1'b1;                           
                end
                `FC1: begin
                    unit1_bound = (cnt_unit1 == 4'd6-1); // no
                    unit2_bound = (cnt_unit2 == 6'd6-1); // no
                    unit3_bound = (cnt_unit3 == 13'd1260-1);
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = 1'b0;
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1);
                    next_weight_idx = unit3_bound? 16'd15540: weight_idx; // next layer
                    next_weight_addr0 = weight_idx + cnt_unit3*2;
                    next_weight_addr1 = weight_idx + cnt_unit3*2 + 1'b1;                           
                end
                `FC2: begin
                    unit1_bound = (cnt_unit1 == 4'd11-1);
                    unit2_bound = (cnt_unit2 == 6'd1-1);
                    unit3_bound = (cnt_unit3 == 13'd110-1); // no
                    next_cnt_unit1 = unit1_bound? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = unit1_bound? (cnt_unit2 + 1'b1): cnt_unit2;
                    next_cnt_unit3 = unit3_bound? 1'b0: (cnt_unit3 + 1'b1);
                    next_weight_idx = unit1_bound? (weight_idx + 16'd21): weight_idx;
                    next_weight_addr0 = weight_idx + cnt_unit1*2;
                    next_weight_addr1 = unit1_bound? (16'd15750 + cnt_unit2): (weight_idx + cnt_unit1*2 + 1'b1);  
                end
                default: begin
                    unit1_bound = 1'b0;
                    unit2_bound = 1'b0;
                    unit3_bound = 1'b0;
                    next_cnt_unit1 = cnt_unit1;
                    next_cnt_unit2 = cnt_unit2;
                    next_cnt_unit3 = cnt_unit3;
                    next_weight_idx = weight_idx;
                    next_weight_addr0 = sram_weight_addr0;
                    next_weight_addr1 = sram_weight_addr1;
                end
            endcase
        end
        else begin
            unit1_bound = 1'b0;
            unit2_bound = 1'b0;
            unit3_bound = 1'b0;
            next_cnt_unit1 = cnt_unit1;
            next_cnt_unit2 = cnt_unit2;
            next_cnt_unit3 = cnt_unit3;
            next_weight_idx = weight_idx;
            next_weight_addr0 = sram_weight_addr0;
            next_weight_addr1 = sram_weight_addr1;
        end
        
        case(state)
            `CONV1, `CONV2: next_weight = {weight[39:0], sram_weight_rdata1[7:0], sram_weight_rdata0};
            `CONV3, `FC1, `FC2, `FINISH: next_weight = {16'b0, sram_weight_rdata1, sram_weight_rdata0};
            default: next_weight = weight;
        endcase
    end
endmodule

//

// mul_iw: several cycles to accumulate the psum
// comparator: 1 cycle for maxpooling
// rqc: 1 cycle for relu, quantization, clamp
module layer (
    input wire clk,
    input wire rst_n,
    input wire [5:0] state, // 2 3-bit
    input wire [1:0] stop, // 2 1-bit
    input wire [10*8-1:0] weight, // 10 8-bit
    input wire [8*8-1:0] in_act, // 8 8-bit
    input wire [9:0] scale,
    output reg [8*8-1:0] out_act, // 8 8-bit
    output reg [7:0] wea,
    output reg do_store,
    output reg layer_done
);

    reg [8:0] cnt_unit1, next_cnt_unit1;
    reg [13:0] cnt_unit2, next_cnt_unit2;
    reg [8*32-1:0] psum; // 8 32-bit
    wire [8*32-1:0] next_psum; // 8 32-bit
    wire [2*32-1:0] max; // 2 32-bit
    wire [2*8-1:0] cur_out_act; // 2 8-bit
    reg [8*8-1:0] next_out_act; // 8 8-bit
    reg [7:0] next_wea;
    reg next_layer_done, next_do_store;

    mul_iw mul1( // next_psum = psum + in_act * weight
        .clk(clk),
        .rst_n(rst_n),
        .state(state[2:0]), // first cycle
        .stop(stop[0]),
        .psum(psum), // 8 32-bit
        .weight(weight),
        .in_act(in_act),
        .next_psum(next_psum) // 8 32-bit
    );
    
    comparator cmp1( // 4 -> 1
        .in({psum[191:128], psum[63:0]}), // 4 32-bit
        .max(max[31:0]) // 32-bit
    ); 

    comparator cmp2( // 4 -> 1
        .in({psum[255:192], psum[127:64]}), // 4 32-bit
        .max(max[63:32]) // 32-bit
    );  

    rqc r1( // relu, quant, clamp
        .in(max[31:0]), // 32-bit
        .scale(scale),
        .out(cur_out_act[7:0]) // 8-bit
    );

    rqc r2( // relu, quant, clamp
        .in(max[63:32]), // 32-bit
        .scale(scale),
        .out(cur_out_act[15:8]) // 8-bit
    );

    always @(posedge clk) begin
        if(!rst_n) begin
            cnt_unit1 <= 1'b0;
            cnt_unit2 <= 1'b0;
            psum <= 1'b0;
            out_act <= 1'b0;
            wea <= 1'b0;
            do_store <= 1'b0;
            layer_done <= 1'b0;
        end
        else begin
            cnt_unit1 <= next_cnt_unit1;
            cnt_unit2 <= next_cnt_unit2;
            psum <= next_psum;
            out_act <= next_out_act;
            wea <= next_wea;
            do_store <= next_do_store;
            layer_done <= next_layer_done;
        end
    end

    always @(*) begin
        next_out_act = out_act;
        if(!stop[1]) begin
            case(state[5:3]) // second cycle
                // [1, 32, 32] -> [6, 14, 14]
                // 6 cycles to compute 2 num, store 8 or 6 num at a time
                // total cycles: 1176/2*6 = 3528
                `CONV1: begin 
                    next_cnt_unit1 = (cnt_unit1 == 9'd42-1)? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = (cnt_unit2 == 14'd3528-1)? 1'b0: (cnt_unit2 + 1'b1);

                    case(cnt_unit1)
                        9'd6-1, 9'd30-1: next_out_act = {48'b0, cur_out_act};
                        9'd12-1, 9'd36-1: next_out_act[31:16] = cur_out_act;
                        9'd18-1, 9'd42-1: next_out_act[47:32] = cur_out_act;
                        9'd24-1: next_out_act[63:48] = cur_out_act;
                    endcase

                    next_do_store = (cnt_unit1 == 9'd24-1) | (cnt_unit1 == 9'd42-1);
                    next_layer_done = (cnt_unit2 == 14'd3528-1);
                    next_wea = 8'b1111_1111;
                end
                // [6, 14, 14] -> [16, 5, 5]
                // 36 cycles to compute 2 or 1 num, 108 cycles to compute 5 num, store 5 num at a time
                // total cycles: 400/5*108 = 8640
                `CONV2: begin
                    next_cnt_unit1 = (cnt_unit1 == 9'd432-1)? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = (cnt_unit2 == 14'd8640-1)? 1'b0: (cnt_unit2 + 1'b1);

                    case(cnt_unit1)
                        9'd36-1: next_out_act = {48'b0, cur_out_act};
                        9'd72-1: next_out_act[31:16] = cur_out_act;
                        9'd108-1: next_out_act[39:32] = cur_out_act[7:0];
                        9'd144-1: next_out_act = {40'b0, cur_out_act, 8'b0};
                        9'd180-1: next_out_act[39:24] = cur_out_act;
                        9'd216-1: next_out_act[47:40] = cur_out_act[7:0];
                        9'd252-1: next_out_act = {32'b0, cur_out_act, 16'b0};
                        9'd288-1: next_out_act[47:32] = cur_out_act;
                        9'd324-1: next_out_act[55:48] = cur_out_act[7:0];
                        9'd360-1: next_out_act = {24'b0, cur_out_act, 24'b0};
                        9'd396-1: next_out_act[55:40] = cur_out_act;
                        9'd432-1: next_out_act[63:56] = cur_out_act[7:0];
                    endcase

                    case(cnt_unit1)
                        9'd108-1: next_wea = 8'b0001_1111;
                        9'd216-1: next_wea = 8'b0011_1110;
                        9'd324-1: next_wea = 8'b0111_1100;
                        9'd432-1: next_wea = 8'b1111_1000;
                        default: next_wea = 8'b0000_0000;
                    endcase
                    
                    next_do_store = (cnt_unit1 == 9'd108-1) | (cnt_unit1 == 9'd216-1) | (cnt_unit1 == 9'd324-1) | (cnt_unit1 == 9'd432-1);
                    next_layer_done = (cnt_unit2 == 14'd8640-1);
                end
                // [16, 5, 5] -> [120, 1, 1]
                // 400/8=50 cycles to compute 1 num, store 8 num at a time
                // total cycles: 120/1*50 = 6000
                `CONV3: begin
                    next_cnt_unit1 = (cnt_unit1 == 9'd400-1)? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = (cnt_unit2 == 14'd6000-1)? 1'b0: (cnt_unit2 + 1'b1);

                    case(cnt_unit1)
                        9'd50-1: next_out_act[7:0] = cur_out_act[7:0];
                        9'd100-1: next_out_act[15:8] = cur_out_act[7:0];
                        9'd150-1: next_out_act[23:16] = cur_out_act[7:0];
                        9'd200-1: next_out_act[31:24] = cur_out_act[7:0];
                        9'd250-1: next_out_act[39:32] = cur_out_act[7:0];
                        9'd300-1: next_out_act[47:40] = cur_out_act[7:0];
                        9'd350-1: next_out_act[55:48] = cur_out_act[7:0];
                        9'd400-1: next_out_act[63:56] = cur_out_act[7:0];
                    endcase

                    next_wea = 8'b1111_1111;
                    next_do_store = (cnt_unit1 == 9'd400-1);
                    next_layer_done = (cnt_unit2 == 14'd6000-1);
                end
                // [120] -> [84]
                // 120/8=15 cycles to compute 1 num, store 8 num at a time (last time is 4)
                // total cycles: 15/1*84 = 1260
                `FC1: begin
                    next_cnt_unit1 = ((cnt_unit1 == 9'd120-1) | (cnt_unit2 == 14'd1260-1))? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = (cnt_unit2 == 14'd1260-1)? 1'b0: (cnt_unit2 + 1'b1);

                    case(cnt_unit1)
                        9'd15-1: next_out_act[7:0] = cur_out_act[7:0];
                        9'd30-1: next_out_act[15:8] = cur_out_act[7:0];
                        9'd45-1: next_out_act[23:16] = cur_out_act[7:0];
                        9'd60-1: next_out_act[31:24] = cur_out_act[7:0];
                        9'd75-1: next_out_act[39:32] = cur_out_act[7:0];
                        9'd90-1: next_out_act[47:40] = cur_out_act[7:0];
                        9'd105-1: next_out_act[55:48] = cur_out_act[7:0];
                        9'd120-1: next_out_act[63:56] = cur_out_act[7:0];
                    endcase

                    next_wea = (cnt_unit2 == 14'd1260-1)? 8'b0000_1111: 8'b1111_1111;
                    next_do_store = (cnt_unit1 == 9'd120-1) | (cnt_unit2 == 14'd1260-1);
                    next_layer_done = (cnt_unit2 == 14'd1260-1);
                end
                // [84] -> [10]
                // 84/8=11 cycles to compute 1 num, store 2 num at a time
                // total cycles: 11/1*10 = 110
                `FC2: begin
                    next_cnt_unit1 = (cnt_unit1 == 9'd22-1)? 1'b0: (cnt_unit1 + 1'b1);
                    next_cnt_unit2 = (cnt_unit2 == 14'd110-1)? 1'b0: (cnt_unit2 + 1'b1);

                    case(cnt_unit1)
                        9'd11-2: next_out_act[31:0] = next_psum[31:0];
                        9'd22-2: next_out_act[63:32] = next_psum[31:0];
                    endcase
                    
                    next_wea = 8'b1111_1111;
                    next_do_store = (cnt_unit1 == 9'd22-2);
                    next_layer_done = (cnt_unit2 == 14'd110-2);
                end
                default: begin
                    next_cnt_unit1 = cnt_unit1;
                    next_cnt_unit2 = cnt_unit2;
                    next_do_store = do_store;
                    next_layer_done = layer_done;
                    next_wea = wea;
                end
            endcase
        end
        else begin
            next_cnt_unit1 = cnt_unit1;
            next_cnt_unit2 = cnt_unit2;
            next_do_store = do_store;
            next_layer_done = layer_done;
            next_wea = wea;
        end
    end
endmodule

// 

module mul_iw(
    input wire clk,
    input wire rst_n,
    input wire [2:0] state,
    input wire stop,
    input wire [8*32-1:0] psum, // 8 32-bit
    input wire [10*8-1:0] weight, // 10 8-bit
    input wire [8*8-1:0] in_act, // 8 8-bit
    output reg [8*32-1:0] next_psum // 8 32-bit
);
    integer i;
    reg [5:0] cnt_unit1, next_cnt_unit1;

    always @(posedge clk) begin
        if(!rst_n) begin
            cnt_unit1 <= 1'b0;
        end
        else begin
            cnt_unit1 <= next_cnt_unit1;
        end
    end
    
    always @(*) begin
        if(!stop) begin
            case(state)
                `CONV1: next_cnt_unit1 = (cnt_unit1 == 6'd6-1)? 1'b0: (cnt_unit1 + 1'b1);
                `CONV2: next_cnt_unit1 = (cnt_unit1 == 6'd36-1)? 1'b0: (cnt_unit1 + 1'b1);
                `CONV3: next_cnt_unit1 = (cnt_unit1 == 6'd50-1)? 1'b0: (cnt_unit1 + 1'b1);
                `FC1: next_cnt_unit1 = (cnt_unit1 == 6'd15-1)? 1'b0: (cnt_unit1 + 1'b1);
                `FC2: next_cnt_unit1 = (cnt_unit1 == 6'd11-1)? 1'b0: (cnt_unit1 + 1'b1);
                default: next_cnt_unit1 = cnt_unit1;
            endcase

            // use paratheses to add two numbers together to reduce area
            case(state)
                `CONV1, `CONV2: begin
                    case(cnt_unit1)
                        6'd0: begin
                            for(i = 0; i < 4; i = i+1) begin
                                next_psum[i*32 +: 32] = ($signed(in_act[i*8 +: 8])*$signed(weight[7:0]) + $signed(in_act[(i+1)*8 +: 8])*$signed(weight[15:8])) + 
                                                        ($signed(in_act[(i+2)*8 +: 8])*$signed(weight[23:16]) + $signed(in_act[(i+3)*8 +: 8])*$signed(weight[31:24])) + 
                                                        $signed(in_act[(i+4)*8 +: 8])*$signed(weight[39:32]);
                            end
                            next_psum[32*8-1:32*4] = 1'b0;
                        end
                        6'd6, 6'd12, 6'd18, 6'd24, 6'd30: begin
                            for(i = 0; i < 4; i = i+1) begin
                                next_psum[i*32 +: 32] = ($signed(in_act[i*8 +: 8])*$signed(weight[7:0]) + $signed(in_act[(i+1)*8 +: 8])*$signed(weight[15:8])) + 
                                                        ($signed(in_act[(i+2)*8 +: 8])*$signed(weight[23:16]) + $signed(in_act[(i+3)*8 +: 8])*$signed(weight[31:24])) + 
                                                        $signed(in_act[(i+4)*8 +: 8])*$signed(weight[39:32]) + $signed(psum[i*32 +: 32]);
                            end
                            next_psum[32*8-1:32*4] = psum[32*8-1:32*4];
                        end
                        6'd5, 6'd11, 6'd17, 6'd23, 6'd29, 6'd35: begin
                            next_psum[32*4-1:0] = psum[32*4-1:0];
                            for(i = 0; i < 4; i = i+1) begin
                                next_psum[(i+4)*32 +: 32] = ($signed(in_act[i*8 +: 8])*$signed(weight[47:40]) + $signed(in_act[(i+1)*8 +: 8])*$signed(weight[55:48])) + 
                                                            ($signed(in_act[(i+2)*8 +: 8])*$signed(weight[63:56]) + $signed(in_act[(i+3)*8 +: 8])*$signed(weight[71:64])) + 
                                                            $signed(in_act[(i+4)*8 +: 8])*$signed(weight[79:72]) + $signed(psum[(i+4)*32 +: 32]);
                            end
                        end
                        default: begin
                            for(i = 0; i < 4; i = i+1) begin
                                next_psum[i*32 +: 32] = ($signed(in_act[i*8 +: 8])*$signed(weight[7:0]) + $signed(in_act[(i+1)*8 +: 8])*$signed(weight[15:8])) + 
                                                        ($signed(in_act[(i+2)*8 +: 8])*$signed(weight[23:16]) + $signed(in_act[(i+3)*8 +: 8])*$signed(weight[31:24])) + 
                                                        $signed(in_act[(i+4)*8 +: 8])*$signed(weight[39:32]) + $signed(psum[i*32 +: 32]);
                            end
                            for(i = 0; i < 4; i = i+1) begin
                                next_psum[(i+4)*32 +: 32] = ($signed(in_act[i*8 +: 8])*$signed(weight[47:40]) + $signed(in_act[(i+1)*8 +: 8])*$signed(weight[55:48])) + 
                                                            ($signed(in_act[(i+2)*8 +: 8])*$signed(weight[63:56]) + $signed(in_act[(i+3)*8 +: 8])*$signed(weight[71:64])) + 
                                                            $signed(in_act[(i+4)*8 +: 8])*$signed(weight[79:72]) + $signed(psum[(i+4)*32 +: 32]);
                            end
                        end
                    endcase
                end
                `CONV3, `FC1: begin
                    next_psum[255:32] = 1'b0;
                    case(cnt_unit1)
                        6'd0: begin
                            next_psum[31:0] = (($signed(in_act[7:0])*$signed(weight[7:0]) + $signed(in_act[15:8])*$signed(weight[15:8])) +
                                            ($signed(in_act[23:16])*$signed(weight[23:16]) + $signed(in_act[31:24])*$signed(weight[31:24]))) + 
                                            (($signed(in_act[39:32])*$signed(weight[39:32]) + $signed(in_act[47:40])*$signed(weight[47:40])) + 
                                            ($signed(in_act[55:48])*$signed(weight[55:48]) + $signed(in_act[63:56])*$signed(weight[63:56])));
                        end
                        default: begin
                            next_psum[31:0] = (($signed(in_act[7:0])*$signed(weight[7:0]) + $signed(in_act[15:8])*$signed(weight[15:8])) +
                                            ($signed(in_act[23:16])*$signed(weight[23:16]) + $signed(in_act[31:24])*$signed(weight[31:24]))) + 
                                            (($signed(in_act[39:32])*$signed(weight[39:32]) + $signed(in_act[47:40])*$signed(weight[47:40])) + 
                                            ($signed(in_act[55:48])*$signed(weight[55:48]) + $signed(in_act[63:56])*$signed(weight[63:56]))) +
                                            $signed(psum[31:0]);
                        end
                    endcase
                end
                `FC2: begin
                    next_psum[255:32] = 1'b0;
                    case(cnt_unit1)
                        6'd0: begin
                            next_psum[31:0] = (($signed(in_act[7:0])*$signed(weight[7:0]) + $signed(in_act[15:8])*$signed(weight[15:8])) +
                                            ($signed(in_act[23:16])*$signed(weight[23:16]) + $signed(in_act[31:24])*$signed(weight[31:24]))) + 
                                            (($signed(in_act[39:32])*$signed(weight[39:32]) + $signed(in_act[47:40])*$signed(weight[47:40])) + 
                                            ($signed(in_act[55:48])*$signed(weight[55:48]) + $signed(in_act[63:56])*$signed(weight[63:56])));
                        end
                        6'd11-1: begin // add bias (weight[63:32])
                            next_psum[31:0] = (($signed(in_act[7:0])*$signed(weight[7:0]) + $signed(in_act[15:8])*$signed(weight[15:8])) +
                                            ($signed(in_act[23:16])*$signed(weight[23:16]) + $signed(in_act[31:24])*$signed(weight[31:24]))) +
                                            $signed(psum[31:0]) + $signed(weight[63:32]);
                        end
                        default: begin
                            next_psum[31:0] = (($signed(in_act[7:0])*$signed(weight[7:0]) + $signed(in_act[15:8])*$signed(weight[15:8])) +
                                            ($signed(in_act[23:16])*$signed(weight[23:16]) + $signed(in_act[31:24])*$signed(weight[31:24]))) + 
                                            (($signed(in_act[39:32])*$signed(weight[39:32]) + $signed(in_act[47:40])*$signed(weight[47:40])) + 
                                            ($signed(in_act[55:48])*$signed(weight[55:48]) + $signed(in_act[63:56])*$signed(weight[63:56]))) +
                                            $signed(psum[31:0]);
                        end
                    endcase
                end
                default: next_psum = psum;
            endcase
        end
        else begin
            next_psum = psum;
            next_cnt_unit1 = cnt_unit1;
        end
    end
endmodule

//

module comparator( // find max
    input wire [4*32-1:0] in, // 4 32-bit
    output reg [31:0] max // 32-bit
);  

    always @(*) begin
        if($signed(in[31:0]) > $signed(in[63:32])) begin
            if($signed(in[95:64]) > $signed(in[127:96]))
                max = ($signed(in[31:0]) > $signed(in[95:64]))? $signed(in[31:0]): $signed(in[95:64]);
            else
                max = ($signed(in[31:0]) > $signed(in[127:96]))? $signed(in[31:0]): $signed(in[127:96]);
        end
        else begin
            if($signed(in[95:64]) > $signed(in[127:96]))
                max = ($signed(in[63:32]) > $signed(in[95:64]))? $signed(in[63:32]): $signed(in[95:64]);
            else
                max = ($signed(in[63:32]) > $signed(in[127:96]))? $signed(in[63:32]): $signed(in[127:96]);
        end
    end

endmodule

//

module rqc( // relu, quant, clamp
    input wire [31:0] in, // 32-bit
    input [9:0] scale,
    output reg [7:0] out // 8-bit
);
    reg [31:0] quant;

    always @(*) begin
        if($signed(in) > 0) begin // relu
            quant = ($signed(scale) * $signed(in)) >>> 16; // quant
            out = (quant > 32'h7f)? 8'h7f: quant[7:0]; // clamp
        end
        else begin
            quant = 32'b0;
            out = 8'b0;
        end
    end

endmodule

//

module store(
    input wire clk,
    input wire rst_n,
    input wire [2:0] state,
    input wire do_store,
    input wire [8*8-1:0] out_act, // 8 8-bit
    input wire [7:0] wea,
    output reg [3:0] sram_act_wea0, sram_act_wea1,
    output reg [9:0] sram_act_addr0, sram_act_addr1,
    output reg [31:0] sram_act_wdata0, sram_act_wdata1
);

    reg [3:0] next_act_wea0, next_act_wea1;
    reg [9:0] next_act_addr0, next_act_addr1;
    reg [31:0] next_sram_act_wdata0, next_sram_act_wdata1;

    always @(posedge clk) begin
        if(!rst_n) begin
            sram_act_wea0 <= 4'b0;
            sram_act_wea1 <= 4'b0;
            sram_act_addr0 <= 16'd0; 
            sram_act_addr1 <= 16'd0;
            sram_act_wdata0 <= 32'b0;
            sram_act_wdata1 <= 32'b0;
        end
        else begin
            sram_act_wea0 <= next_act_wea0;
            sram_act_wea1 <= next_act_wea1;
            sram_act_addr0 <= next_act_addr0; 
            sram_act_addr1 <= next_act_addr1; 
            sram_act_wdata0 <= next_sram_act_wdata0;
            sram_act_wdata1 <= next_sram_act_wdata1;
        end
    end

    always @(*) begin
        if(do_store) begin
            next_act_wea0 = wea[3:0];
            next_act_wea1 = wea[7:4]; 
            next_sram_act_wdata0 = out_act[31:0];
            next_sram_act_wdata1 = out_act[63:32];
            case(state)
                `CONV1, `CONV3, `FC1: begin                 
                    next_act_addr0 = sram_act_addr0 + 2'd2;
                    next_act_addr1 = sram_act_addr1 + 2'd2;      
                end
                `CONV2: begin
                    case(wea) 
                        8'b0001_1111: begin
                            next_act_addr0 = sram_act_addr0 + 2'd2;
                            next_act_addr1 = sram_act_addr1 + 2'd2; 
                        end
                        default: begin
                            next_act_addr0 = sram_act_addr0 + 2'd1;
                            next_act_addr1 = sram_act_addr1 + 2'd1; 
                        end
                    endcase                   
                end
                `FC2: begin
                    if(sram_act_addr0 == 16'd742) begin // because last time only use one                  
                        next_act_addr0 = sram_act_addr0 + 2'd1;
                        next_act_addr1 = sram_act_addr1 + 2'd1; 
                    end
                    else begin
                        next_act_addr0 = sram_act_addr0 + 2'd2;
                        next_act_addr1 = sram_act_addr1 + 2'd2; 
                    end
                end
                default: begin                   
                    next_act_addr0 = sram_act_addr0;
                    next_act_addr1 = sram_act_addr1; 
                end
            endcase
        end
        else begin
            next_act_wea0 = 4'b0;
            next_act_wea1 = 4'b0;
            next_sram_act_wdata0 = 1'b0;
            next_sram_act_wdata1 = 1'b0;
            next_act_addr0 = (state == `WAIT)? 16'd256-2: sram_act_addr0;
            next_act_addr1 = (state == `WAIT)? 16'd257-2: sram_act_addr1;
        end
    end

endmodule