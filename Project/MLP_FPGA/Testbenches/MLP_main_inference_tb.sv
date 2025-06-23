//======================================================================
// Testbench for the MLP Accelerator
//======================================================================
// This testbench verifies the functionality of the 'mlp' module.
//
// Test Strategy:
// 1.  Read input vectors and weight matrices from text files.
// 2.  Execute a bit-accurate behavioral model (software model) in SystemVerilog to calculate the expected output.
// 3.  Configure the DUT (Device Under Test) by writing the same inputs and weights to its memory-mapped registers.
// 4.  Start the DUT's computation and poll for the 'done' flag.
// 5.  Read the result from the DUT.
// 6.  Compare the DUT's output with the expected output from the behavioral model.
// 7.  Report PASS or FAIL.
//======================================================================

`timescale 1ns / 100ps

module mlp_inference_tb;

    //--- Testbench Parameters ---
    // DUT configuration parameters, must match the DUT's parameters.
    localparam N_INPUTS     = 2;
    localparam N_HIDDEN     = 4;
    localparam N_OUTPUT     = 1;
    localparam IN_WIDTH     = 16;
    localparam WGT_WIDTH    = 16;
    localparam MAC_WIDTH    = 32;           // Internal accumulator precision, must match DUT
    localparam OUT_WIDTH    = 16;           // Final and intermediate output width

    // Clock period for the simulation (10ns -> 100MHz)
    localparam CLK_PERIOD = 10;

    // Max positive value for a signed OUT_WIDTH number
    localparam signed [OUT_WIDTH-1:0] MAX_SIGNED_OUT_VAL = (1 << (OUT_WIDTH-1)) - 1;

    //--- DUT Register Map ---
    // Addresses and control bit positions for the DUT's memory-mapped interface.
    localparam CTRL_REG_ADDR    = 2'd0;
    localparam INPUT_FIFO_ADDR  = 2'd1;
    localparam WEIGHT_FIFO_ADDR = 2'd2;
    localparam OUTPUT_REG_ADDR  = 2'd3;

    localparam CTRL_RUN_BIT_POS    = 0;
    localparam CTRL_DONE_BIT_POS   = 1;
	localparam CTRL_INTERRUPT_BIT_POS = 2;  // Interrupt enable bit, not used in this test
    localparam CTRL_LAYER_SEL_BIT_POS = 3;  // Selects which layer's weights to write

    //--- Testbench Signals ---
    // These signals connect to the DUT's ports.
    reg clk, rst;
    reg write_en;
    reg [1:0] addr;
    reg [31:0] writedata;                   // Data to write to the DUT
    wire [31:0] readdata;                   // Data read from the DUT
    wire irq;

    //--- DUT Instantiation ---
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

    //--- Clock Generation ---
    always #(CLK_PERIOD/2) clk = ~clk;

    //--- Helper Task: Write Register ---
    // This task simplifies writing a value to a DUT register over two clock cycles.
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

    //--- Behavioral Model Storage ---
    // These arrays store the inputs, weights, and expected result for the software model.
    reg signed [IN_WIDTH-1:0]   tb_inputs [0:N_INPUTS-1];
    reg signed [WGT_WIDTH-1:0]  tb_hidden_weights [0:N_HIDDEN-1][0:N_INPUTS];
    reg signed [WGT_WIDTH-1:0]  tb_output_weights [0:N_OUTPUT-1][0:N_HIDDEN];
    reg signed [OUT_WIDTH-1:0]  tb_expected_output_value;



    //======================================================================
    // Behavioral Model Functions
    //======================================================================

    //--- ReLU and Saturation Function ---
    // This function models the neuron's activation (ReLU) and saturation (clipping) stage.
    // It must exactly match the hardware's behavior.
    function automatic signed [OUT_WIDTH-1:0] apply_relu_saturate_func (input signed [MAC_WIDTH-1:0] sum_val);
        if (sum_val < 0) begin
            return 0;                       // ReLU: max(0, x)
        end else if (sum_val > MAX_SIGNED_OUT_VAL) begin
            return MAX_SIGNED_OUT_VAL;      // Saturate to max positive for signed OUT_WIDTH
        end else begin
            return sum_val;                  // Implicit truncation if MAC_WIDTH > OUT_WIDTH
        end
    endfunction : apply_relu_saturate_func
    
    //--- Main Behavioral Model Function ---
    // This function performs a full MLP forward pass using fixed-point arithmetic
    // to produce a bit-accurate expected result.
    function automatic signed [OUT_WIDTH-1:0] calculate_mlp_behavioral (
        input signed [IN_WIDTH-1:0]   p_inputs[0:N_INPUTS-1],
        input signed [WGT_WIDTH-1:0]  p_hidden_weights[0:N_HIDDEN-1][0:N_INPUTS],
        input signed [WGT_WIDTH-1:0]  p_output_weights[0:N_OUTPUT-1][0:N_HIDDEN]
    );
        // Internal variables for calculation
        logic signed [MAC_WIDTH-1:0]  l_hidden_sum[0:N_HIDDEN-1];
        logic signed [MAC_WIDTH-1:0]  l_hidden_scaled_sum[0:N_HIDDEN-1];    // Scaled sum
        logic signed [OUT_WIDTH-1:0]  l_hidden_activated[0:N_HIDDEN-1];
        logic signed [MAC_WIDTH-1:0]  l_output_sum; 
        logic signed [MAC_WIDTH-1:0]  l_output_scaled_sum;                  // Scaled sum
        logic signed [OUT_WIDTH-1:0]  l_final_output;
        integer i, j;

        $display("Behavioral Func: Max signed %0d-bit output value for ReLU: %0d", OUT_WIDTH, MAX_SIGNED_OUT_VAL);

        $display("Behavioral Func: Calculating Hidden Layer Outputs (with ReLU)...");
        for (i = 0; i < N_HIDDEN; i = i + 1) begin
            // Pre-scale the bias term to match the scale of the multiplication results.
            l_hidden_sum[i] = p_hidden_weights[i][0] <<< WGT_WIDTH/2;       // Bias
            // Accumulate the product of inputs and weights
            for (j = 0; j < N_INPUTS; j = j + 1) begin
                l_hidden_sum[i] = l_hidden_sum[i] + p_inputs[j] * p_hidden_weights[i][j+1];
            end
            // Scale the final sum back down by shifting right (fixed-point division).
            l_hidden_scaled_sum[i] = l_hidden_sum[i] >>> (WGT_WIDTH/2);
            // Apply activation and saturation.
            l_hidden_activated[i] = apply_relu_saturate_func(l_hidden_scaled_sum[i]);
            $display("  Func Hidden Neuron %0d: Bias=%0d (%h), Sum = %0d, Scaled Sum = %0d, ReLU Output = %0d", i, p_hidden_weights[i][0], p_hidden_weights[i][0], l_hidden_sum[i], l_hidden_scaled_sum[i], l_hidden_activated[i]);
        end

        $display("Behavioral Func: Calculating Output Layer Outputs (with ReLU)...");
        // Pre-scale bias for the output layer.
        l_output_sum = p_output_weights[0][0] << WGT_WIDTH/2;               // Bias
        for (j = 0; j < N_HIDDEN; j = j + 1) begin
            // Accumulate products of hidden layer outputs and output layer weights.
            l_output_sum = l_output_sum + l_hidden_activated[j] * p_output_weights[0][j+1];
            $display(" l_output_sum after adding hidden neuron %0d output: %0d, p_output_weights[0][%0d] = %0d, l_hidden_activated[%0d] = %0d", j, l_output_sum, j+1, p_output_weights[0][j+1], j, l_hidden_activated[j]);
        end
        // Scale down the final output sum.
        l_output_scaled_sum = l_output_sum >>> (WGT_WIDTH/2);
        // The final output does not have ReLU in this model.
        l_final_output = l_output_scaled_sum;
        $display("  Func Output Neuron 0: Bias=%0d (%h), Sum = %0d, Scaled Sum = %0d, Decimal number = %0f, Decimal number divided again = %0f", p_output_weights[0][0],p_output_weights[0][0],l_output_sum,l_output_scaled_sum, l_output_sum/256.0 , l_output_scaled_sum/256.0);

        return l_final_output;
    endfunction : calculate_mlp_behavioral



    //======================================================================
    // Main Test Sequence
    //======================================================================
    initial begin
        logic signed [OUT_WIDTH-1:0] dut_output_value;
        logic [31:0] current_ctrl_reg_val;                                  // For polling DONE bit

       // Temporary 1D arrays for reading flattened weight matrices from files.
        logic [WGT_WIDTH-1:0] temp_hidden_weights_1d [0 : N_HIDDEN * (N_INPUTS + 1) - 1];
        logic [WGT_WIDTH-1:0] temp_output_weights_1d [0 : N_OUTPUT * (N_HIDDEN + 1) - 1];
        integer k;                                                          // Index for 1D array

        //--- Setup and Reset ---
        $dumpfile("mlp.vcd");
        $dumpvars(0, mlp_inference_tb);

        clk = 0;
        rst = 1;
        write_en = 0;
        addr = CTRL_REG_ADDR;                                               // Initialize addr
        writedata = 0;

        repeat(2) @(posedge clk);
        rst = 0;
        @(posedge clk);                                                     // Allow a cycle for reset to propagate fully

        //--- Data Preparation: Load weights and define inputs ---
        // Define the input vector for this test case.
        tb_inputs[0] = (-1*(1 << IN_WIDTH/2)); 
        tb_inputs[1] = (2*(1 << IN_WIDTH/2));
        $display("TB: Input vector x = [%0d, %0d]", tb_inputs[0], tb_inputs[1]);

        // Read hidden layer weights from file into a temporary 1D array.
        $display("[%0t] Main TB: Reading hidden layer weights from weights_w1.txt...", $time);
        $readmemb("weights_w1.txt", temp_hidden_weights_1d);
        
        // Map the 1D data into the 2D array used by the behavioral model.
        k = 0;
        for (int h_idx = 0; h_idx < N_HIDDEN; h_idx = h_idx + 1) begin
        tb_hidden_weights[h_idx][0] = temp_hidden_weights_1d[k++];          // Bias
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

        // Read output layer weights from file.
        $display("[%0t] Main TB: Reading output layer weights from weights_w2.txt...", $time);
        $readmemb("weights_w2.txt", temp_output_weights_1d);

        // Map to 2D array.
        k = 0;
        for (int o_idx = 0; o_idx < N_OUTPUT; o_idx = o_idx + 1) begin      // N_OUTPUT is 1
            tb_output_weights[o_idx][0] = temp_output_weights_1d[k++];      // Bias
            for (int h_w_idx = 0; h_w_idx < N_HIDDEN; h_w_idx = h_w_idx + 1) begin
                tb_output_weights[o_idx][h_w_idx+1] = temp_output_weights_1d[k++];
            end
        end
        $display("[%0t] Main TB: Finished reading and mapping output layer weights.", $time);


        //--- Behavioral Model Execution ---
        // Calculate the expected "golden" result before running the DUT.
        tb_expected_output_value = calculate_mlp_behavioral(tb_inputs, tb_hidden_weights, tb_output_weights);
        $display("[%0t] Main TB: Expected output from behavioral function = %0d (%f)", $time, tb_expected_output_value, tb_expected_output_value / 256.0);



        //======================================================================
        // DUT Configuration and Execution
        //======================================================================

        // --- Load Input Vector into DUT ---
        write_reg(INPUT_FIFO_ADDR, tb_inputs[0]);
        write_reg(INPUT_FIFO_ADDR, tb_inputs[1]);

        // --- Load Hidden Layer Weights into DUT ---
        // The control register defaults to selecting the hidden layer (layer 0).
        for (int h_idx = 0; h_idx < N_HIDDEN; h_idx = h_idx + 1) begin
            write_reg(WEIGHT_FIFO_ADDR, tb_hidden_weights[h_idx][0]); // Bias
            for (int i_idx = 0; i_idx < N_INPUTS; i_idx = i_idx + 1) begin
                write_reg(WEIGHT_FIFO_ADDR, tb_hidden_weights[h_idx][i_idx+1]);
            end
        end

        // --- Load Output Layer Weights into DUT ---
        $display("[%0t] Main TB: Loading output layer weights into DUT...", $time);
        // First, select the output layer (layer 1) for weight writing.
        // We can safely overwrite the whole control register because we are in the IDLE state,
        // so the RUN and DONE bits are guaranteed to be 0.
        write_reg(CTRL_REG_ADDR, (1 << CTRL_LAYER_SEL_BIT_POS));            // Select output layer for weights

        // Now, write the weights.
        for (int o_idx = 0; o_idx < N_OUTPUT; o_idx = o_idx + 1) begin 
            write_reg(WEIGHT_FIFO_ADDR, tb_output_weights[o_idx][0]);       // Bias
            for (int h_w_idx = 0; h_w_idx < N_HIDDEN; h_w_idx = h_w_idx + 1) begin
                write_reg(WEIGHT_FIFO_ADDR, tb_output_weights[o_idx][h_w_idx+1]);
            end
        end

        // --- Start MLP Computation ---
        // Write a '1' to the RUN bit to start the FSM.
        write_reg(CTRL_REG_ADDR, (1 << CTRL_RUN_BIT_POS));

        // --- Poll for Completion ---
        // Wait until the DUT asserts the DONE bit in the control register.
        $display("[%0t] Main TB: Waiting for MLP computation DONE...", $time);
        @(posedge clk); 
        current_ctrl_reg_val = readdata;                                    // Read initial CTRL value
        while (current_ctrl_reg_val[CTRL_DONE_BIT_POS] == 0) begin
            @(posedge clk);
            current_ctrl_reg_val = readdata;                                // Keep reading until DONE is high
        end
        $display("[%0t] Main TB: MLP computation DONE. CTRL_REG = 0x%0h", $time, current_ctrl_reg_val);
        
        // === Read output ===
        write_reg(OUTPUT_REG_ADDR, 32'd0);                                            // Set address to read output
        @(posedge clk); 
        dut_output_value = readdata;                                        // Capture the registered output data

        $display("[%0t] Main TB: MLP output from DUT: %0d (raw readdata: 0x%0h), Decimal = %0f", $time, dut_output_value, readdata, dut_output_value / 256.0);
        $display("[%0t] Main TB: MLP expected output (behavioral model): %0d, Decimal = %0f", $time, tb_expected_output_value, tb_expected_output_value / 256.0);



        //======================================================================
        // Verification and Reporting
        //======================================================================
        if (dut_output_value == tb_expected_output_value) begin
            $display(">>> TEST PASSED: DUT output matches behavioral model.");
        end else begin
            $error(">>> TEST FAILED: DUT output %0d does not match expected %0d.",
                   dut_output_value, tb_expected_output_value);
        end

        $finish;

    end

endmodule