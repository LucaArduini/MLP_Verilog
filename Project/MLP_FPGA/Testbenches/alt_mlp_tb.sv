`timescale 1ns / 100ps

module alt_mlp_tb;

    // Parameters
    localparam N_INPUTS     = 2;
    localparam N_HIDDEN     = 4;
    localparam N_OUTPUT     = 1;
    localparam IN_WIDTH     = 16;
    localparam WGT_WIDTH    = 16;
    localparam MAC_WIDTH    = 32; // For internal accumulation precision
    localparam OUT_WIDTH    = 16; // Final output and intermediate activated output width

    localparam CLK_PERIOD = 10;

    // Max positive value for a signed OUT_WIDTH number
    localparam signed [OUT_WIDTH-1:0] MAX_SIGNED_OUT_VAL = (1 << (OUT_WIDTH-1)) - 1;

    // DUT Register Addresses and Control Bits
    localparam CTRL_REG_ADDR    = 2'd0;
    localparam INPUT_FIFO_ADDR  = 2'd1;
    localparam WEIGHT_FIFO_ADDR = 2'd2;
    localparam OUTPUT_REG_ADDR  = 2'd3;

    localparam CTRL_RUN_BIT_POS    = 0;
    localparam CTRL_DONE_BIT_POS   = 1;
    localparam CTRL_LAYER_SEL_BIT_POS = 3;


    // Testbench signals
    reg clk, rst;
    reg write_en;
    reg [1:0] addr;
    reg [31:0] writedata; // DUT register write data
    wire [31:0] readdata; // DUT register read data
    wire irq;

    // Instantiate DUT
    mlp #(
        .N_INPUTS(N_INPUTS),
        .N_HIDDEN(N_HIDDEN),
        .N_OUTPUT(N_OUTPUT),
        .IN_WIDTH(IN_WIDTH),
        .WGT_WIDTH(WGT_WIDTH),
        .MAC_WIDTH(MAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .write_en(write_en),
        .addr(addr),
        .writedata(writedata),
        .readdata(readdata),
        .irq(irq)
    );

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // Task to write a register
    task write_reg(input [1:0] a, input [31:0] data);
        begin
            @(posedge clk);
            write_en = 1;
            addr = a;
            writedata = data;
            @(posedge clk);
            write_en = 0;
        end
    endtask

    // Behavioral Model Storage
    reg signed [IN_WIDTH-1:0]   tb_inputs [0:N_INPUTS-1];
    reg signed [WGT_WIDTH-1:0]  tb_hidden_weights [0:N_HIDDEN-1][0:N_INPUTS];
    reg signed [WGT_WIDTH-1:0]  tb_output_weights [0:N_OUTPUT-1][0:N_HIDDEN];
    reg signed [OUT_WIDTH-1:0]  tb_expected_output_value;


    // --- Behavioral Model Functions ---
    function automatic signed [OUT_WIDTH-1:0] apply_relu_saturate_func (input signed [MAC_WIDTH-1:0] sum_val);
        if (sum_val < 0) begin
            return 0; // ReLU: max(0, x)
        end else if (sum_val > MAX_SIGNED_OUT_VAL) begin
            return MAX_SIGNED_OUT_VAL; // Saturate to max positive for signed OUT_WIDTH
        end else begin
            return sum_val; // Implicit truncation if MAC_WIDTH > OUT_WIDTH
        end
    endfunction : apply_relu_saturate_func
    

    function automatic signed [OUT_WIDTH-1:0] calculate_mlp_behavioral (
        input signed [IN_WIDTH-1:0]   p_inputs[0:N_INPUTS-1],
        input signed [WGT_WIDTH-1:0]  p_hidden_weights[0:N_HIDDEN-1][0:N_INPUTS],
        input signed [WGT_WIDTH-1:0]  p_output_weights[0:N_OUTPUT-1][0:N_HIDDEN]
    );
        logic signed [MAC_WIDTH-1:0]  l_hidden_sum[0:N_HIDDEN-1];
        logic signed [MAC_WIDTH-1:0]  l_hidden_scaled_sum[0:N_HIDDEN-1]; // Scaled sum
        logic signed [OUT_WIDTH-1:0]  l_hidden_activated[0:N_HIDDEN-1];
        logic signed [MAC_WIDTH-1:0]  l_output_sum; 
        logic signed [MAC_WIDTH-1:0]  l_output_scaled_sum; // Scaled sum
        logic signed [OUT_WIDTH-1:0]  l_final_output;
        integer i, j;

        $display("Behavioral Func: Max signed %0d-bit output value for ReLU: %0d", OUT_WIDTH, MAX_SIGNED_OUT_VAL);

        $display("Behavioral Func: Calculating Hidden Layer Outputs (with ReLU)...");
        for (i = 0; i < N_HIDDEN; i = i + 1) begin
            // Calculate the weighted sum for each hidden neuron
            l_hidden_sum[i] = p_hidden_weights[i][0] <<< WGT_WIDTH/2; // Bias
            for (j = 0; j < N_INPUTS; j = j + 1) begin
                l_hidden_sum[i] = l_hidden_sum[i] + p_inputs[j] * p_hidden_weights[i][j+1];
            end
            l_hidden_scaled_sum[i] = l_hidden_sum[i] >>> WGT_WIDTH/2; // Arithmetic right shift
            l_hidden_activated[i] = apply_relu_saturate_func(l_hidden_scaled_sum[i]);
            $display("  Func Hidden Neuron %0d: Bias=%0d (%h), Sum = %0d, Scaled Sum = %0d, ReLU Output = %0d", i, p_hidden_weights[i][0], p_hidden_weights[i][0], l_hidden_sum[i], l_hidden_scaled_sum[i], l_hidden_activated[i]);
        end

        $display("Behavioral Func: Calculating Output Layer Outputs (with ReLU)...");
        l_output_sum = p_output_weights[0][0] << WGT_WIDTH/2; // Bias
        for (j = 0; j < N_HIDDEN; j = j + 1) begin
            //p_output_weights[0][j+1] = p_output_weights[0][j+1] <<< WGT_WIDTH/2; // Scale weights
            l_output_sum = l_output_sum + l_hidden_activated[j] * p_output_weights[0][j+1];
            $display(" l_output_sum after adding hidden neuron %0d output: %0d, p_output_weights[0][%0d] = %0d, l_hidden_activated[%0d] = %0d", j, l_output_sum, j+1, p_output_weights[0][j+1], j, l_hidden_activated[j]);
        end
        l_output_scaled_sum = l_output_sum >>> WGT_WIDTH/2; // Arithmetic right shift
        l_final_output = l_output_scaled_sum;
        $display("  Func Output Neuron 0: Bias=%0d (%h), Sum = %0d, Scaled Sum = %0d, Decimal number = %0f, Decimal number divided again = %0f", p_output_weights[0][0],p_output_weights[0][0],l_output_sum,l_output_scaled_sum, l_output_sum/256.0 , l_output_scaled_sum/256.0);

        return l_final_output;
    endfunction : calculate_mlp_behavioral


    // Initial stimulus
    initial begin
        logic signed [OUT_WIDTH-1:0] dut_output_value;
        logic [31:0] current_ctrl_reg_val; // For polling DONE bit

        // Temporary 1D arrays for $readmemb
        logic [WGT_WIDTH-1:0] temp_hidden_weights_1d [0 : N_HIDDEN * (N_INPUTS + 1) - 1];
        logic [WGT_WIDTH-1:0] temp_output_weights_1d [0 : N_OUTPUT * (N_HIDDEN + 1) - 1];
        integer k; // Index for 1D array

        $dumpfile("mlp.vcd");
        $dumpvars(0, alt_mlp_tb);

        clk = 0;
        rst = 1;
        write_en = 0;
        addr = CTRL_REG_ADDR; // Initialize addr
        writedata = 0;

        repeat(2) @(posedge clk);
        rst = 0;
        @(posedge clk); // Allow a cycle for reset to propagate fully

        // --- Populate Behavioral Model Inputs ---
        tb_inputs[0] = (-1*(1 << IN_WIDTH/2)); 
        tb_inputs[1] = (2*(1 << IN_WIDTH/2));
        $display("TB: Input vector x = [%0d, %0d]", tb_inputs[0], tb_inputs[1]);
        // --- Read Hidden Layer Weights from File using $readmemb ---
        $display("[%0t] Main TB: Reading hidden layer weights from weights_w1.txt...", $time);
        $readmemb("weights_w1.txt", temp_hidden_weights_1d);
        
        // Copy from 1D temp array to 2D tb_hidden_weights
        k = 0;
        for (int h_idx = 0; h_idx < N_HIDDEN; h_idx = h_idx + 1) begin
            tb_hidden_weights[h_idx][0] = temp_hidden_weights_1d[k++]; // Bias
            for (int i_idx = 0; i_idx < N_INPUTS; i_idx = i_idx + 1) begin
                tb_hidden_weights[h_idx][i_idx+1] = temp_hidden_weights_1d[k++];
            end
        end
        $display("[%0t] Main TB: Finished reading and mapping hidden layer weights.", $time);
        // For verification, display some loaded weights
        for (int h_idx = 0; h_idx < N_HIDDEN; h_idx = h_idx + 1) begin
            $display("TB: Hidden Neuron %0d Weights (Hex): Bias=%h, W_in0=%h, W_in1=%h", h_idx, tb_hidden_weights[h_idx][0], tb_hidden_weights[h_idx][1], tb_hidden_weights[h_idx][2]);
            $display("TB: Hidden Neuron %0d Weights (Dec): Bias=%d, W_in0=%d, W_in1=%d", h_idx, tb_hidden_weights[h_idx][0], tb_hidden_weights[h_idx][1], tb_hidden_weights[h_idx][2]);
        end


        // --- Read Output Layer Weights from File using $readmemb ---
        $display("[%0t] Main TB: Reading output layer weights from weights_w2.txt...", $time);
        $readmemb("weights_w2.txt", temp_output_weights_1d);

        // Copy from 1D temp array to 2D tb_output_weights
        k = 0;
        for (int o_idx = 0; o_idx < N_OUTPUT; o_idx = o_idx + 1) begin // N_OUTPUT is 1
            tb_output_weights[o_idx][0] = temp_output_weights_1d[k++]; // Bias
            for (int h_w_idx = 0; h_w_idx < N_HIDDEN; h_w_idx = h_w_idx + 1) begin
                tb_output_weights[o_idx][h_w_idx+1] = temp_output_weights_1d[k++];
            end
        end
        $display("[%0t] Main TB: Finished reading and mapping output layer weights.", $time);
        // For verification:
        // $display("TB: Output Neuron 0 Weights (Hex): Bias=%h, W_h0=%h, W_h1=%h, W_h2=%h, W_h3=%h", tb_output_weights[0][0], tb_output_weights[0][1], tb_output_weights[0][2], tb_output_weights[0][3], tb_output_weights[0][4]);
        // $display("TB: Output Neuron 0 Weights (Dec): Bias=%d, W_h0=%d, W_h1=%d, W_h2=%d, W_h3=%d", tb_output_weights[0][0], tb_output_weights[0][1], tb_output_weights[0][2], tb_output_weights[0][3], tb_output_weights[0][4]);


        // --- Calculate Expected Output using Behavioral Function ---
        tb_expected_output_value = calculate_mlp_behavioral(tb_inputs, tb_hidden_weights, tb_output_weights);
        $display("[%0t] Main TB: Expected output from behavioral function = %0d, Decimal = %0f", $time, tb_expected_output_value, tb_expected_output_value / 256.0);

        // --- DUT Configuration and Execution ---
        // === Load input vector: x = [7, -3] ===
        write_reg(INPUT_FIFO_ADDR, tb_inputs[0]); // Use tb_inputs values
        write_reg(INPUT_FIFO_ADDR, tb_inputs[1]);

        // === Load hidden layer weights into DUT ===
        $display("[%0t] Main TB: Loading hidden layer weights into DUT...", $time);
        for (int h_idx = 0; h_idx < N_HIDDEN; h_idx = h_idx + 1) begin
            write_reg(WEIGHT_FIFO_ADDR, tb_hidden_weights[h_idx][0]); // Bias
            for (int i_idx = 0; i_idx < N_INPUTS; i_idx = i_idx + 1) begin
                write_reg(WEIGHT_FIFO_ADDR, tb_hidden_weights[h_idx][i_idx+1]);
            end
        end

        // === Load output layer weights into DUT ===
        $display("[%0t] Main TB: Loading output layer weights into DUT...", $time);
        write_reg(CTRL_REG_ADDR, (1 << CTRL_LAYER_SEL_BIT_POS)); // Select output layer for weights
        for (int o_idx = 0; o_idx < N_OUTPUT; o_idx = o_idx + 1) begin 
            write_reg(WEIGHT_FIFO_ADDR, tb_output_weights[o_idx][0]); // Bias
            for (int h_w_idx = 0; h_w_idx < N_HIDDEN; h_w_idx = h_w_idx + 1) begin
                write_reg(WEIGHT_FIFO_ADDR, tb_output_weights[o_idx][h_w_idx+1]);
            end
        end

        // === Start MLP computation ===
        write_reg(CTRL_REG_ADDR, (1 << CTRL_RUN_BIT_POS));

        // Wait for DONE 
        $display("[%0t] Main TB: Waiting for MLP computation DONE...", $time);
        @(posedge clk); 
        current_ctrl_reg_val = readdata;
        while (current_ctrl_reg_val[CTRL_DONE_BIT_POS] == 0) begin
            @(posedge clk);
            current_ctrl_reg_val = readdata;
        end
        $display("[%0t] Main TB: MLP computation DONE. CTRL_REG = 0x%0h", $time, current_ctrl_reg_val);
        
        // === Read output ===
        write_reg(OUTPUT_REG_ADDR, 32'd0); 
        @(posedge clk); 
        dut_output_value = readdata;

        $display("[%0t] Main TB: MLP output from DUT: %0d (raw readdata: 0x%0h), Decimal = %0f", $time, dut_output_value, readdata, dut_output_value / 256.0);
        $display("[%0t] Main TB: MLP expected output (behavioral model): %0d, Decimal = %0f", $time, tb_expected_output_value, tb_expected_output_value / 256.0);

        // === Compare and Report ===
        if (dut_output_value == tb_expected_output_value) begin
            $display(">>> TEST PASSED: DUT output matches behavioral model.");
        end else begin
            $error(">>> TEST FAILED: DUT output %0d does not match expected %0f. Raw DUT readdata: 0x%0h",
                   dut_output_value, tb_expected_output_value, readdata);
        end

        $finish;
    end

endmodule