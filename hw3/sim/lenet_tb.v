`timescale 1ns/1ps
`define CYCLE 12
`define END_CYCLES 50000
module tb_lenet();

    

    // ===== System Signals =====
    reg clk;
    integer i, cycle_count;
    reg start_count;


    // ===== SRAM Signals =====
    wire [ 3:0] sram_weight_wea0;
    wire [15:0] sram_weight_addr0;
    wire [31:0] sram_weight_wdata0;
    wire [31:0] sram_weight_rdata0;
    wire [ 3:0] sram_weight_wea1;
    wire [15:0] sram_weight_addr1;
    wire [31:0] sram_weight_wdata1;
    wire [31:0] sram_weight_rdata1;
    
    wire [ 3:0] sram_act_wea0;
    wire [15:0] sram_act_addr0;
    wire [31:0] sram_act_wdata0;
    wire [31:0] sram_act_rdata0;
    wire [ 3:0] sram_act_wea1;
    wire [15:0] sram_act_addr1;
    wire [31:0] sram_act_wdata1;
    wire [31:0] sram_act_rdata1;

    // ===== Golden =====
    reg [31:0] golden [0:1023];

    // ===== Lenet Signals =====
    reg rst_n;
    reg compute_start;
    wire compute_finish;
    
    reg [31:0] scale_CONV1;
    reg [31:0] scale_CONV2;
    reg [31:0] scale_CONV3;
    reg [31:0] scale_FC1;
    reg [31:0] scale_FC2;

    // ===== Module instantiation =====
    lenet lenet_inst(
        .clk(clk),
        .rst_n(rst_n),

        .compute_start(compute_start),
        .compute_finish(compute_finish),

        // Quantization scale
        .scale_CONV1(scale_CONV1),
        .scale_CONV2(scale_CONV2),
        .scale_CONV3(scale_CONV3),
        .scale_FC1(scale_FC1),
        .scale_FC2(scale_FC2),

        // weight sram, single port
        .sram_weight_wea0(sram_weight_wea0),
        .sram_weight_addr0(sram_weight_addr0),
        .sram_weight_wdata0(sram_weight_wdata0),
        .sram_weight_rdata0(sram_weight_rdata0),
        .sram_weight_wea1(sram_weight_wea1),
        .sram_weight_addr1(sram_weight_addr1),
        .sram_weight_wdata1(sram_weight_wdata1),
        .sram_weight_rdata1(sram_weight_rdata1),

        // Output sram,dual port
        .sram_act_wea0(sram_act_wea0),
        .sram_act_addr0(sram_act_addr0),
        .sram_act_wdata0(sram_act_wdata0),
        .sram_act_rdata0(sram_act_rdata0),
        .sram_act_wea1(sram_act_wea1),
        .sram_act_addr1(sram_act_addr1),
        .sram_act_wdata1(sram_act_wdata1),
        .sram_act_rdata1(sram_act_rdata1)
    );

    SRAM_weight_16384x32b weight_sram( 
        .clk(clk),
        .wea0(sram_weight_wea0),
        .addr0(sram_weight_addr0),
        .wdata0(sram_weight_wdata0),
        .rdata0(sram_weight_rdata0),
        .wea1(sram_weight_wea1),
        .addr1(sram_weight_addr1),
        .wdata1(sram_weight_wdata1),
        .rdata1(sram_weight_rdata1)
    );
    
    SRAM_activation_1024x32b act_sram( 
        .clk(clk),
        .wea0(sram_act_wea0),
        .addr0(sram_act_addr0),
        .wdata0(sram_act_wdata0),
        .rdata0(sram_act_rdata0),
        .wea1(sram_act_wea1),
        .addr1(sram_act_addr1),
        .wdata1(sram_act_wdata1),
        .rdata1(sram_act_rdata1)
    );



    // ===== Load data ===== //
    initial begin
        weight_sram.load_data("../pattern/weights/weights.csv");
        act_sram.load_data("../pattern/patterns/image00.csv");
        $readmemh("../pattern/patterns/golden00.csv", golden);

    end


    // ===== System reset ===== //
    initial begin
        clk = 0;
        rst_n = 1;
        compute_start = 0;
        cycle_count = 0;
    end
    
    // ===== Cycle count ===== //
    initial begin
        wait(compute_start == 1);
        start_count = 1;
        wait(compute_finish == 1);
        start_count = 0;
    end

    always @(posedge clk) begin
        if(start_count)
            cycle_count <= cycle_count + 1;
    end 
   
    // ===== Time Exceed Abortion ===== //
    initial begin
        #(`CYCLE*`END_CYCLES);
        $display("\n========================================================");
        $display("You have exceeded the cycle count limit.");
        $display("Simulation abort");
        $display("========================================================");
        $finish;    
    end

    // ===== Clk fliping ===== //
    always #(`CYCLE/2) begin
        clk = ~clk;
    end 

    // ===== Set simulation info ===== //
    initial begin
    `ifdef GATESIM
        $fsdbDumpfile("lenet_syn.fsdb");
        $fsdbDumpvars;
        $sdf_annotate("../syn/netlist/lenet_syn.sdf", lenet_inst);
	`else
        `ifdef POSTSIM
            $fsdbDumpfile("lenet_post.fsdb");
            $fsdbDumpvars;
            $sdf_annotate("../apr/netlist/CHIP.sdf", lenet_inst);
        `else
            $fsdbDumpfile("lenet.fsdb");
            $fsdbDumpvars;
        `endif
    `endif
    end
        

    // ===== Simulating  ===== //
    initial begin

        scale_CONV1 = 91;
        scale_CONV2 = 257;
        scale_CONV3 = 287;
        scale_FC1 = 427;
        scale_FC2 = 321;

        #(`CYCLE*100);
        $display("Reset System");
        @(negedge clk);
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        $display("Compute start");
        @(negedge clk);
        compute_start = 1'b1;
        @(negedge clk);
        compute_start = 1'b0;

        wait(compute_finish == 1);
        $display("Compute finished, start validating result...");

        validate();

        $display("Simulation finish");
        $finish;
    end

    integer errors, total_errors;
    task validate; begin
        // Input Image
        
        total_errors = 0;
        $display("=====================");

        errors = 0;
        for(i=0 ; i<256 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR] Image Result:%8h Golden:%8h", act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end
            else begin
                //$display("[CORRECT]   [%d] Image Result:%8h Golden:%8h", i, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("Image             [PASS]");
        else
            $display("Image             [FAIL]");
        total_errors = total_errors + errors;
            
        errors = 0;
        for(i=256 ; i<592 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR]   [%d] Conv1 Result:%8h Golden:%8h", i-256, act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end
            else begin
                //$display("[CORRECT]   [%d] Conv1 Result:%8h Golden:%8h", i-256, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("Conv 1 activation [PASS]");
        else
            $display("Conv 1 activation [FAIL]");
        total_errors = total_errors + errors;
            
        errors = 0;
        for(i=592 ; i<692 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR]     [%d] Conv2 Result:%8h Golden:%8h", i-592, act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end
            else begin
                //$display("[CORRECT]   [%d] Conv2 Result:%8h Golden:%8h", i-592, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("Conv 2 activation [PASS]");
        else
            $display("Conv 2 activation [FAIL]");
        total_errors = total_errors + errors;

        
        errors = 0;
        for(i=692 ; i<722 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR]     [%d] Conv3 Result:%8h Golden:%8h", i-692, act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end
            else begin
                //$display("[CORRECT]   [%d] Conv2 Result:%8h Golden:%8h", i-692, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("Conv 3 activation [PASS]");
        else
            $display("Conv 3 activation [FAIL]");
        total_errors = total_errors + errors;
        
        errors = 0;
        for(i=722 ; i<743 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                //$display("[ERROR] FC1 Result:%8h Golden:%8h", act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end 
            else begin
                //$display("[CORRECT]   [%d] FC1 Result:%8h Golden:%8h", i-722, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("FC   1 activation [PASS]");
        else 
            $display("FC   1 activation [FAIL]");
        total_errors = total_errors + errors;
        
        errors = 0;
        for(i=743 ; i<753 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR] FC2 Result:%8h Golden:%8h", act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end 
            else begin
                //$display("[CORRECT]   [%d] FC2 Result:%8h Golden:%8h", i-743, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("FC   2 activation [PASS]");
        else
            $display("FC   2 activation [FAIL]");
        total_errors = total_errors + errors;

        
        if(total_errors == 0)
            $display(">>> Congratulation! All result are correct");
        else
            $display(">>> There are %d errors QQ", total_errors);
            
    `ifdef GATESIM
        $display("  [Pre-layout gate-level simulation]");
	`else
        `ifdef POSTSIM
            $display("  [Post-layout gate-level simulation]");
        `else
            $display("  [RTL simulation]");
        `endif
    `endif
        $display("Clock Period: %.2f ns,Total cycle count: %d cycles", `CYCLE, cycle_count);
        $display("=====================");
    end
    endtask



endmodule
