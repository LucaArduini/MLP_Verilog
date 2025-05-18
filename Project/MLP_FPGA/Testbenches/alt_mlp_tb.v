`timescale 1ns / 100ps

module alt_mlp_tb;

    // Parameters
    localparam N_INPUTS     = 2;
    localparam N_HIDDEN     = 4;
    localparam N_OUTPUT     = 1;
    localparam IN_WIDTH     = 16;
    localparam WGT_WIDTH    = 16;
    localparam MAC_WIDTH    = 64; // For internal accumulation precision
    localparam OUT_WIDTH    = 16; // Final output and intermediate activated output width

    localparam CLK_PERIOD = 10;

    // Max positive value for a signed OUT_WIDTH number
    localparam signed [OUT_WIDTH-1:0] MAX_SIGNED_OUT_VAL = (1 << (OUT_WIDTH-1)) - 1;
    // Min negative value (not used by ReLU, but good to know)
    // localparam signed [OUT_WIDTH-1:0] MIN_SIGNED_OUT_VAL = -(1 << (OUT_WIDTH-1));


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
    // Hidden weights: [neuron_idx][weight_idx], where weight_idx=0 is bias, 1..N_INPUTS are input weights
    reg signed [WGT_WIDTH-1:0]  tb_hidden_weights [0:N_HIDDEN-1][0:N_INPUTS];
    // Output weights: [neuron_idx][weight_idx], where weight_idx=0 is bias, 1..N_HIDDEN are hidden layer output weights
    reg signed [WGT_WIDTH-1:0]  tb_output_weights [0:N_OUTPUT-1][0:N_HIDDEN];

    // Expected output storage (since N_OUTPUT=1, a single variable is fine, but an array is more general)
    reg signed [OUT_WIDTH-1:0]  tb_expected_output_value;


    // --- Behavioral Model Functions ---

    // Function to apply ReLU and saturate/truncate to OUT_WIDTH
    function automatic signed [OUT_WIDTH-1:0] apply_relu_saturate_func (input signed [MAC_WIDTH-1:0] sum_val);
        if (sum_val < 0) begin
            return 0; // ReLU: max(0, x)
        end else if (sum_val > MAX_SIGNED_OUT_VAL) begin
            return MAX_SIGNED_OUT_VAL; // Saturate to max positive for signed OUT_WIDTH
        end else begin
            // Value is positive and fits within OUT_WIDTH range (or will be truncated correctly by assignment)
            return sum_val; // Implicit truncation if MAC_WIDTH > OUT_WIDTH
        end
    endfunction : apply_relu_saturate_func

    // Local storage for intermediate values
    

    // SystemVerilog function for behavioral MLP calculation
    // This version is specific to N_OUTPUT=1 for simplicity of return type.
    // For a general N_OUTPUT, it would return an array/queue or take an output array argument.
    function automatic signed [OUT_WIDTH-1:0] calculate_mlp_behavioral (
        input signed [IN_WIDTH-1:0]   p_inputs[0:N_INPUTS-1],
        input signed [WGT_WIDTH-1:0]  p_hidden_weights[0:N_HIDDEN-1][0:N_INPUTS],
        input signed [WGT_WIDTH-1:0]  p_output_weights[0:N_OUTPUT-1][0:N_HIDDEN]
    );
        logic signed [MAC_WIDTH-1:0]  l_hidden_sum[0:N_HIDDEN-1];
        logic signed [OUT_WIDTH-1:0]  l_hidden_activated[0:N_HIDDEN-1];
        logic signed [MAC_WIDTH-1:0]  l_output_sum; // Since N_OUTPUT=1 for this function version
        logic signed [OUT_WIDTH-1:0]  l_final_output;
        integer i, j;

        $display("Behavioral Func: Max signed %0d-bit output value for ReLU: %0d", OUT_WIDTH, MAX_SIGNED_OUT_VAL);

        // 1. Hidden Layer Calculation
        $display("Behavioral Func: Calculating Hidden Layer Outputs (with ReLU)...");
        for (i = 0; i < N_HIDDEN; i = i + 1) begin
            l_hidden_sum[i] = p_hidden_weights[i][0]; // Start with bias
            for (j = 0; j < N_INPUTS; j = j + 1) begin
                l_hidden_sum[i] = l_hidden_sum[i] + p_inputs[j] * p_hidden_weights[i][j+1];
            end
            l_hidden_activated[i] = apply_relu_saturate_func(l_hidden_sum[i]);
            $display("  Func Hidden Neuron %0d: Sum = %0d, ReLU Output = %0d", i, l_hidden_sum[i], l_hidden_activated[i]);
        end

        // 2. Output Layer Calculation
        $display("Behavioral Func: Calculating Output Layer Outputs (with ReLU)...");
        l_output_sum = p_output_weights[0][0]; // Bias for the first (and only) output neuron
        for (j = 0; j < N_HIDDEN; j = j + 1) begin
            l_output_sum = l_output_sum + l_hidden_activated[j] * p_output_weights[0][j+1];
        end
        l_final_output = apply_relu_saturate_func(l_output_sum);
        $display("  Func Output Neuron 0: Sum = %0d, Expected ReLU Output = %0d", l_output_sum, l_final_output);

        return l_final_output;
    endfunction : calculate_mlp_behavioral


    // Initial stimulus
    initial begin
        logic signed [OUT_WIDTH-1:0] dut_output_value;

        $dumpfile("mlp.vcd");
        $dumpvars(0, alt_mlp_tb);

        clk = 0;
        rst = 1;
        write_en = 0;
        addr = 0;
        writedata = 0;

        repeat(2) @(posedge clk);
        rst = 0;

        // --- Populate Behavioral Model Inputs & Weights ---
        // Inputs: x = [7, -3]
        tb_inputs[0] = 7;
        tb_inputs[1] = -3;

        // Hidden Layer Weights (Format: [bias, w_for_x0, w_for_x1])
        tb_hidden_weights[0][0] = 1;  tb_hidden_weights[0][1] = 2;  tb_hidden_weights[0][2] = 3;  // N0
        tb_hidden_weights[1][0] = 0;  tb_hidden_weights[1][1] = -1; tb_hidden_weights[1][2] = 2;  // N1
        tb_hidden_weights[2][0] = -2; tb_hidden_weights[2][1] = 4;  tb_hidden_weights[2][2] = 1;  // N2
        tb_hidden_weights[3][0] = 1;  tb_hidden_weights[3][1] = 1;  tb_hidden_weights[3][2] = 1;  // N3

        // Output Layer Weights (Format: [bias, w_for_h0, w_for_h1, w_for_h2, w_for_h3])
        // N_OUTPUT = 1, so tb_output_weights[0] is used.
        tb_output_weights[0][0] = 1; // Bias for output neuron 0
        tb_output_weights[0][1] = 1; // Weight for h0
        tb_output_weights[0][2] = 1; // Weight for h1
        tb_output_weights[0][3] = 1; // Weight for h2
        tb_output_weights[0][4] = 1; // Weight for h3

        // --- Calculate Expected Output using Behavioral Function ---
        tb_expected_output_value = calculate_mlp_behavioral(tb_inputs, tb_hidden_weights, tb_output_weights);
        $display("Main TB: Expected output from behavioral function = %0d", tb_expected_output_value);

        // --- DUT Configuration and Execution ---
        // === Load input vector: x = [7, -3] ===
        write_reg(2'd1, 32'sd7);
        write_reg(2'd1, -32'sd3);

        // === Load hidden layer weights ===
        // Neuron 0
        write_reg(2'd2, 32'sd1); write_reg(2'd2, 32'sd2); write_reg(2'd2, 32'sd3);
        // Neuron 1
        write_reg(2'd2, 32'sd0); write_reg(2'd2, -32'sd1); write_reg(2'd2, 32'sd2);
        // Neuron 2
        write_reg(2'd2, -32'sd2); write_reg(2'd2, 32'sd4); write_reg(2'd2, 32'sd1);
        // Neuron 3
        write_reg(2'd2, 32'sd1); write_reg(2'd2, 32'sd1); write_reg(2'd2, 32'sd1);

        // === Load output layer weights ===
        write_reg(2'd0, 32'sb1000); // change layer
        write_reg(2'd2, 32'sd1); // bias
        write_reg(2'd2, 32'sd1); // w_h0
        write_reg(2'd2, 32'sd1); // w_h1
        write_reg(2'd2, 32'sd1); // w_h2
        write_reg(2'd2, 32'sd1); // w_h3

        // === Start MLP computation ===
        write_reg(2'd0, 32'b00000001);  // Set RUN bit

        // Wait for DONE (bit 1 in CTRL)
        @(posedge clk);
        wait(dut.ctrl[1] == 1);
        $display("MLP computation DONE.");
        write_reg(2'd3, 32'd0);
        // === Read output ===
        @(posedge clk); // Allow one cycle for readdata to update

        
        // Assuming DUT sign-extends its OUT_WIDTH result into the 32-bit readdata.
        // Since ReLU output is >=0, the upper bits of readdata should be 0.
        dut_output_value = readdata;

        $display("MLP output from DUT: %0d (raw readdata: 0x%0h)", dut_output_value, readdata);
        $display("MLP expected output (behavioral model): %0d", tb_expected_output_value);

        // === Compare and Report ===
        if (dut_output_value == tb_expected_output_value) begin
            $display(">>> TEST PASSED: DUT output matches behavioral model.");
        end else begin
            $error(">>> TEST FAILED: DUT output %0d does not match expected %0d",
                   dut_output_value, tb_expected_output_value);
        end

        $finish;
    end

endmodule