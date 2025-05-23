// Ensure MLP_mac.v and MLP_weight_mem.v are available for compilation
// `include "MLP_weight_mem.v" // Optional, depending on simulator/flow
// `include "MLP_mac.v"        // Optional, depending on simulator/flow

`timescale 1ns/1ps

module tb_MLP_layer_hidden;

    // Parameters for the MLP_layer_hidden (matching DUT defaults or configurable)
    localparam N_INPUTS     = 2+1;
    localparam N_NEURONS    = 4;
    localparam IN_WIDTH     = 16;
    localparam WGT_WIDTH    = 16;
    localparam MAC_WIDTH    = 64;       // Width of MAC accumulator
    localparam OUT_WIDTH    = 16;       // Width of each output element

    localparam CLK_PERIOD = 10; // ns

    // Testbench signals
    logic clk;
    // Weight memory write interface
    logic                               wr_en_tb;
    logic   [WGT_WIDTH-1:0]             wr_weight_tb;
    logic   [$clog2(N_NEURONS)-1:0]     wr_row_tb;    // Neuron index
    logic   [$clog2(N_INPUTS)-1:0]      wr_col_tb;    // Input feature index for weight

    // Input and control signals
    logic   signed [IN_WIDTH-1:0]       input_value_tb; // Current input feature value
    logic   [$clog2(N_INPUTS)-1:0]      input_index_tb; // Index of current input feature
    logic                               valid_tb;       // Triggers MAC accumulation
    logic                               start_tb;       // Clears/initializes MAC accumulators
    logic                               relu_en_tb;     // Triggers output registration

    // Output vector from DUT
    wire [N_NEURONS*OUT_WIDTH-1:0]      outputs_flat_dut;

    // Instantiate the Device Under Test (DUT)
    MLP_layer_hidden #(
        .N_INPUTS     (N_INPUTS),
        .N_NEURONS    (N_NEURONS),
        .IN_WIDTH     (IN_WIDTH),
        .WGT_WIDTH    (WGT_WIDTH),
        .MAC_WIDTH    (MAC_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) dut (
        .clk(clk),

        .wr_en      (wr_en_tb),
        .wr_weight  (wr_weight_tb),
        .wr_row     (wr_row_tb),
        .wr_col     (wr_col_tb),

        .input_value (input_value_tb),
        .input_index (input_index_tb),
        .valid      (valid_tb),
        .start      (start_tb),
        .relu_en    (relu_en_tb),

        .outputs_flat (outputs_flat_dut)
    );

    // Clock generator
    always # (CLK_PERIOD/2) clk = ~clk;

    // Storage for test data (weights and inputs)
    logic signed [N_NEURONS-1:0][N_INPUTS-1:0][WGT_WIDTH-1:0] current_test_weights;
    logic signed [N_INPUTS-1:0][IN_WIDTH-1:0]                 current_test_input_vector;

    logic signed [IN_WIDTH-1:0]   val_i_16b;
    logic signed [WGT_WIDTH-1:0]  val_w_16b;

    // --- Helper function to calculate expected output ---
    // This function models the DUT's computation path (MAC, ReLU, Clip)
    function automatic logic [N_NEURONS*OUT_WIDTH-1:0] calculate_expected_output_behavioral(
        logic signed [N_INPUTS-1:0][IN_WIDTH-1:0]     inputs,
        logic signed [N_NEURONS-1:0][N_INPUTS-1:0][WGT_WIDTH-1:0] weights
    );
        logic signed [MAC_WIDTH-1:0] neuron_mac_sum [N_NEURONS-1:0];
        logic [N_NEURONS*OUT_WIDTH-1:0] expected_flat_output;
        logic signed [OUT_WIDTH-1:0] temp_out_val;
        const logic signed [OUT_WIDTH-1:0] MAX_VAL_CALC = (1 << (OUT_WIDTH-1)) - 1;

        $display("--- Behavioral Model Calculation ---");
        for (int n = 0; n < N_NEURONS; n++) begin
            neuron_mac_sum[n] = 0;
            // $display("BM: Neuron %0d, Initial sum = %d", n, neuron_mac_sum[n]); // Optional
            for (int i = 0; i < N_INPUTS; i++) begin
                logic signed [IN_WIDTH+WGT_WIDTH-1:0] product_temp;
                logic signed [MAC_WIDTH-1:0] product_extended;

                // Check the actual weight and input values being used by the function
                if (n < 2) begin // Only print for first few neurons to avoid excessive output initially
                    $display("BM: n=%0d, i=%0d, input_val_fn_arg[%0d]=%d (0x%h), weight_fn_arg[%0d][%0d]=%d (0x%h)",
                            n, i, i, inputs[i], inputs[i], n, i, weights[n][i], weights[n][i]);
                end

                
                val_i_16b = inputs[i];
                val_w_16b = weights[n][i];

                // Explicitly perform signed multiplication.
                // The result of (signed [A-1:0] * signed [B-1:0]) is signed [A+B-1:0].
                // So, product_temp should be sized like this: (signed [IN_WIDTH+WGT_WIDTH-1:0])
                product_temp = val_i_16b * val_w_16b;

                // Sign-extend product to MAC_WIDTH.
                if ( (IN_WIDTH + WGT_WIDTH) > MAC_WIDTH ) begin
                     $error("TB: Product width (%0d) > MAC_WIDTH (%0d). Mismatch with MLP_mac assumption.", IN_WIDTH+WGT_WIDTH, MAC_WIDTH);
                     product_extended = product_temp[MAC_WIDTH-1:0];
                end else if ( (IN_WIDTH + WGT_WIDTH) < MAC_WIDTH) begin // Explicitly handle extension
                     product_extended = {{(MAC_WIDTH - (IN_WIDTH+WGT_WIDTH)){product_temp[IN_WIDTH+WGT_WIDTH-1]}}, product_temp};
                end else begin // (IN_WIDTH + WGT_WIDTH) == MAC_WIDTH
                     product_extended = product_temp;
                end
                neuron_mac_sum[n] = neuron_mac_sum[n] + product_extended;

                if (n < 2) begin // Only print for first few neurons
                    $display("BM: n=%0d, i=%0d, product_temp=%d, product_extended=%d, neuron_mac_sum[%0d]=%d (0x%h)",
                             n, i, product_temp, product_extended, n, neuron_mac_sum[n], neuron_mac_sum[n]);
                end
            end

            if (n < 2) begin // Only print for first few neurons
                 $display("BM: Neuron %0d FINAL MAC SUM = %d (0x%h)", n, neuron_mac_sum[n], neuron_mac_sum[n]);
            end

            // Apply ReLU and clipping
            
            if (neuron_mac_sum[n] < 0) begin
                temp_out_val = 0;
            end else if (neuron_mac_sum[n] > MAX_VAL_CALC) begin
                temp_out_val = MAX_VAL_CALC;
            end else begin
                temp_out_val = neuron_mac_sum[n][OUT_WIDTH-1:0];
            end
            expected_flat_output[(n*OUT_WIDTH) +: OUT_WIDTH] = temp_out_val;
            if (n < 2) begin // Only print for first few neurons
                 $display("BM: Neuron %0d Output Val = %d (0x%h)", n, temp_out_val, temp_out_val);
            end
        end
        $display("--- End Behavioral Model Calculation ---");
        return expected_flat_output;
    endfunction

    logic [N_NEURONS*OUT_WIDTH-1:0] expected_outputs_tc1;
    logic [N_NEURONS*OUT_WIDTH-1:0] expected_outputs_tc2;
    logic [N_NEURONS*OUT_WIDTH-1:0] expected_outputs_tc3;
    // --- Test Sequence ---
    initial begin
        $display("[%0t] Starting Testbench for MLP_layer_hidden", $time);
        clk = 1; // Initialize clk, first transition will be to 0

        // Initialize all control signals to a known benign state
        wr_en_tb       = 1'b0;
        wr_weight_tb   = {WGT_WIDTH{1'b0}};
        wr_row_tb      = {$clog2(N_NEURONS){1'b0}};
        wr_col_tb      = {$clog2(N_INPUTS){1'b0}};
        input_value_tb = {IN_WIDTH{1'b0}};
        input_index_tb = {$clog2(N_INPUTS){1'b0}};
        valid_tb       = 1'b0;
        start_tb       = 1'b0;
        relu_en_tb     = 1'b0;

        repeat(2) @(posedge clk); // Wait for a couple of clock cycles

        // --- Test Case 1: Load weights and process one input vector ---
        $display("[%0t] Test Case 1: Load weights and process first input vector.", $time);

        // Define sample weights (Neuron Index x Input Feature Index)
        for (int n = 0; n < N_NEURONS; n++) begin
            for (int i = 0; i < N_INPUTS; i++) begin
                current_test_weights[n][i] = signed'(n + i + 1 + (n*2) - (i*3)); // Pattern for varied small weights
            end
        end
        // Override specific weights for targeted testing if needed
        // if (N_NEURONS >= 2 && N_INPUTS >=1 ) current_test_weights[1][0] = -5;         // Negative weight
        // if (N_NEURONS >=3 && WGT_WIDTH == 16) current_test_weights[2][0] = 1000;     // Larger weight

        // Load weights into the DUT's weight memories
        wr_en_tb = 1'b1;
        for (int n = 0; n < N_NEURONS; n++) begin // Iterate over neuron rows
            for (int i = 0; i < N_INPUTS; i++) begin // Iterate over input columns (weights per neuron)
                wr_row_tb = n;
                wr_col_tb = i;
                wr_weight_tb = current_test_weights[n][i];
                @(posedge clk);
                // $display("[%0t] Loaded weight W_n%0d_i%0d = %d", $time, n, i, current_test_weights[n][i]);
            end
        end
        wr_en_tb = 1'b0; // De-assert write enable
        // Clear write address signals for safety, though not strictly necessary if wr_en=0
        wr_row_tb      = {$clog2(N_NEURONS){1'b0}}; 
        wr_col_tb      = {$clog2(N_INPUTS){1'b0}};
        @(posedge clk); // Ensure wr_en=0 is seen before MAC operations start

        // Define a sample input vector
        for (int i = 0; i < N_INPUTS; i++) begin
            current_test_input_vector[i] = signed'(i + 1 + (i* -1)); // e.g., [1, 0, -1, -2] for N_INPUTS=4
        end
        if (N_INPUTS >= 2) current_test_input_vector[1] = -2; // Specific value


        // Process the input vector through the DUT
        $display("[%0t] Processing input vector (TC1): ", $time);
        for(int i=0; i < N_INPUTS; i++) $write("%d ", current_test_input_vector[i]);
        $write("\n");

        // Cycle 1: Assert 'start' for the first input feature to initialize/load MACs
        start_tb = 1'b1;
        valid_tb = 1'b1; // Often 'valid' is also asserted with 'start'
        input_value_tb = current_test_input_vector[0];
        input_index_tb = 0;
        @(posedge clk);
        // $display("[%0t] MAC Start: input_idx=%0d, input_val=%d", $time, input_index_tb, input_value_tb);

        // Cycles 2 to N_INPUTS: Assert 'valid' for subsequent input features to accumulate
        start_tb = 1'b0; // De-assert start after the first feature
        for (int i = 1; i < N_INPUTS; i++) begin
            valid_tb = 1'b1;
            input_value_tb = current_test_input_vector[i];
            input_index_tb = i;
            @(posedge clk);
            // $display("[%0t] MAC Valid: input_idx=%0d, input_val=%d", $time, input_index_tb, input_value_tb);
        end

        // Cycle N_INPUTS+1: All features processed. De-assert valid. Assert 'relu_en' to trigger output update.
        valid_tb = 1'b0;
        relu_en_tb = 1'b1;
        // input_index_tb/input_value_tb don't strictly matter here but set to benign values
        input_index_tb = {$clog2(N_INPUTS){1'b0}}; 
        input_value_tb = {IN_WIDTH{1'b0}};
        @(posedge clk);
        $display("[%0t] ReLU Enabled (TC1). outputs_flat should now be valid.", $time);
        @(posedge clk);
        // Calculate expected output using the behavioral model
        
        expected_outputs_tc1 = calculate_expected_output_behavioral(current_test_input_vector, current_test_weights);

        // Check results
        if (outputs_flat_dut === expected_outputs_tc1) begin
            $display("[%0t] PASS: Test Case 1 output matches expected.", $time);
            $display("[%0t] Expected: %h, Got: %h", $time, expected_outputs_tc1, outputs_flat_dut);
        end else begin
            $error("[%0t] FAIL: Test Case 1 output MISMATCH.", $time);
            $display("[%0t] Expected: %h", $time, expected_outputs_tc1);
            $display("[%0t] Got     : %h", $time, outputs_flat_dut);
        end
        
        relu_en_tb = 1'b0; // De-assert relu_en
        @(posedge clk);    // Allow relu_en=0 to propagate

        @(posedge clk);
        // --- Test Case 2: Process a second, different input vector with the SAME weights ---
        // This verifies that the `start` signal correctly resets/re-initializes the MAC accumulators.
        $display("[%0t] Test Case 2: Process second input vector (tests MAC reset with existing weights).", $time);
        
        for (int i = 0; i < N_INPUTS; i++) begin
            current_test_input_vector[i] = (i % 2 == 0) ? signed'(5 + i) : signed'(-5 - i); // e.g., [5, -6, 7, -8]
        end
        $display("[%0t] Processing input vector (TC2): ", $time);
        for(int i=0; i < N_INPUTS; i++) $write("%d ", current_test_input_vector[i]);
        $write("\n");

        // Process the input vector (same sequence of control signals as TC1)
        start_tb = 1'b1; valid_tb = 1'b1;
        input_value_tb = current_test_input_vector[0]; input_index_tb = 0;
        @(posedge clk);
        
        start_tb = 1'b0;
        for (int i = 1; i < N_INPUTS; i++) begin
            valid_tb = 1'b1;
            input_value_tb = current_test_input_vector[i]; input_index_tb = i;
            @(posedge clk);
        end
        
        valid_tb = 1'b0; relu_en_tb = 1'b1;
        @(posedge clk);
        $display("[%0t] ReLU Enabled (TC2). outputs_flat should now be valid.", $time);

        @(posedge clk);
        expected_outputs_tc2 = calculate_expected_output_behavioral(current_test_input_vector, current_test_weights);

        if (outputs_flat_dut === expected_outputs_tc2) begin
            $display("[%0t] PASS: Test Case 2 output matches expected.", $time);
            $display("[%0t] Expected: %h, Got: %h", $time, expected_outputs_tc2, outputs_flat_dut);
        end else begin
            $error("[%0t] FAIL: Test Case 2 output MISMATCH.", $time);
            $display("[%0t] Expected: %h", $time, expected_outputs_tc2);
            $display("[%0t] Got     : %h", $time, outputs_flat_dut);
        end
        
        relu_en_tb = 1'b0;
        @(posedge clk);

        @(posedge clk);
        // --- Test Case 3: Load NEW weights and process an input vector ---
        // This verifies that the weight memories can be updated.
        $display("[%0t] NEW Test Case 3: Load NEW weights and process an input vector.", $time);

        for (int n = 0; n < N_NEURONS; n++) begin
            for (int i = 0; i < N_INPUTS; i++) begin
                current_test_weights[n][i] = signed'((n * N_INPUTS + i) * (( (n+i)%2==0 ) ? 1 : -1) * 20 + 10); // New pattern
                 if (N_NEURONS > 0 && N_INPUTS > 0 && n==0 && i==0 && WGT_WIDTH==16 && OUT_WIDTH==16) begin
                    // Setup to test clipping for neuron 0 if inputs are e.g. all 10
                    current_test_weights[0][0] = 1000; 
                    if (N_INPUTS > 1) current_test_weights[0][1] = 1000;
                    if (N_INPUTS > 2) current_test_weights[0][2] = 1000;
                    if (N_INPUTS > 3) current_test_weights[0][3] = 1000; 
                 end
                 if (N_NEURONS > 1 && N_INPUTS > 0 && n==1 && i==0 && WGT_WIDTH==16) begin
                    // Setup to test ReLU to zero for neuron 1
                    current_test_weights[1][0] = -2000;
                    if (N_INPUTS > 1) current_test_weights[1][1] = -1; // small neg to keep sum neg
                 end
            end
        end

        wr_en_tb = 1'b1;
        for (int n = 0; n < N_NEURONS; n++) begin
            for (int i = 0; i < N_INPUTS; i++) begin
                wr_row_tb = n; wr_col_tb = i; wr_weight_tb = current_test_weights[n][i];
                @(posedge clk);
            end
        end
        wr_en_tb = 1'b0;
        @(posedge clk);

        for (int i = 0; i < N_INPUTS; i++) begin
            current_test_input_vector[i] = 10; // Input vector of all 10s
        end
        $display("[%0t] Processing input vector (TC3 - new weights): ", $time);
        for(int i=0; i < N_INPUTS; i++) $write("%d ", current_test_input_vector[i]);
        $write("\n");

        // Process input vector
        start_tb = 1'b1; valid_tb = 1'b1;
        input_value_tb = current_test_input_vector[0]; input_index_tb = 0;
        @(posedge clk);
        
        start_tb = 1'b0;
        for (int i = 1; i < N_INPUTS; i++) begin
            valid_tb = 1'b1;
            input_value_tb = current_test_input_vector[i]; input_index_tb = i;
            @(posedge clk);
        end
        
        valid_tb = 1'b0; relu_en_tb = 1'b1;
        @(posedge clk);
        $display("[%0t] ReLU Enabled (TC3). outputs_flat should now be valid.", $time);

        @(posedge clk);
        expected_outputs_tc3 = calculate_expected_output_behavioral(current_test_input_vector, current_test_weights);

        if (outputs_flat_dut === expected_outputs_tc3) begin
            $display("[%0t] PASS: Test Case 3 output matches expected.", $time);
            $display("[%0t] Expected: %h, Got: %h", $time, expected_outputs_tc3, outputs_flat_dut);
        end else begin
            $error("[%0t] FAIL: Test Case 3 output MISMATCH.", $time);
            $display("[%0t] Expected: %h", $time, expected_outputs_tc3);
            $display("[%0t] Got     : %h", $time, outputs_flat_dut);
        end
        
        relu_en_tb = 1'b0;
        @(posedge clk);
        @(posedge clk);

        $display("[%0t] All tests completed.", $time);
        $finish;
    end

    // Optional: Monitor signals for debugging (can be very verbose)
    // initial begin
    //     $monitor("[%0tns] clk=%b, wr_en=%b, wr_r=%d, wr_c=%d, wr_w=%d | start=%b, valid=%b, in_idx=%d, in_val=%d | relu=%b | out_flat=%h",
    //              $time, clk, wr_en_tb, wr_row_tb, wr_col_tb, wr_weight_tb,
    //              start_tb, valid_tb, input_index_tb, input_value_tb, relu_en_tb, outputs_flat_dut);
    // end

endmodule