`timescale 1ns / 100ps

module mlp_driver;

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

    integer       mcd_input_file;   // File descriptor for input.txt
    reg [1023:0]  line_buffer;      // Buffer for $fgets (e.g., for up to 128 characters)
    integer       char_count;       // Characters read by $fgets
    integer       num_parsed;       // Items parsed by $sscanf
    integer       driver_inputs[0:1]; // Array to store the two inputs


    


    // Initial stimulus
    initial begin
        logic signed [OUT_WIDTH-1:0] dut_output_value;
        logic [31:0] current_ctrl_reg_val; // For polling DONE bit

        // Temporary 1D arrays for $readmemb
        logic [WGT_WIDTH-1:0] temp_hidden_weights_1d [0 : N_HIDDEN * (N_INPUTS + 1) - 1];
        logic [WGT_WIDTH-1:0] temp_output_weights_1d [0 : N_OUTPUT * (N_HIDDEN + 1) - 1];
        integer k; // Index for 1D array

        clk = 0;
        rst = 1;
        write_en = 0;
        addr = CTRL_REG_ADDR; // Initialize addr
        writedata = 0;

        repeat(2) @(posedge clk);
        rst = 0;
        @(posedge clk); // Allow a cycle for reset to propagate fully

        // --- This is the part "added" to make it read from input.txt ---
        // Open the file "input.txt" for reading.
        mcd_input_file = $fopen("input.txt", "r");

        // Basic check to ensure the file was opened successfully.
        // Your snippet has more detailed error handling for read operations.
        if (mcd_input_file == 0) begin
            $error("[%0t] Testbench: Failed to open 'input.txt'. Aborting.", $time);
            $finish;
        end
        // --- End of added part ---

        // --------------------------------------------------------------------
        // --- THE FOLLOWING CODE IS EXACTLY AS PROVIDED IN YOUR REQUEST ---
        // --------------------------------------------------------------------

        // Read first input from file
        if (!$feof(mcd_input_file)) begin
            char_count = $fgets(line_buffer, mcd_input_file);
            if (char_count > 0) begin // Check if anything was read
                num_parsed = $sscanf(line_buffer, "%d", tb_inputs[0]);
                if (num_parsed != 1) begin
                    $error("[%0t] MLP Driver: Failed to parse first input from 'input.txt'. Line content: \"%s\". Aborting.", $time, line_buffer);
                    $fclose(mcd_input_file);
                    $finish;
                end
            end else if (!$feof(mcd_input_file)) begin // Read error if not EOF
                 $error("[%0t] MLP Driver: Error reading first input line from 'input.txt'. Aborting.", $time);
                 $fclose(mcd_input_file);
                 $finish;
            end else begin // Premature EOF
                 $error("[%0t] MLP Driver: Premature end-of-file in 'input.txt' when expecting first input. Aborting.", $time);
                 $fclose(mcd_input_file);
                 $finish;
            end
        end else begin
            $error("[%0t] MLP Driver: 'input.txt' is empty or could not be read for the first input. Aborting.", $time);
            $fclose(mcd_input_file); // Close even if initial check fails, though mcd_input_file would be 0 then.
            $finish;
        end

        // Read second input from file
        if (!$feof(mcd_input_file)) begin
            char_count = $fgets(line_buffer, mcd_input_file);
             if (char_count > 0) begin
                num_parsed = $sscanf(line_buffer, "%d", tb_inputs[1]);
                if (num_parsed != 1) begin
                    $error("[%0t] MLP Driver: Failed to parse second input from 'input.txt'. Line content: \"%s\". Aborting.", $time, line_buffer);
                    $fclose(mcd_input_file);
                    $finish;
                end
            end else if (!$feof(mcd_input_file)) begin
                 $error("[%0t] MLP Driver: Error reading second input line from 'input.txt'. Aborting.", $time);
                 $fclose(mcd_input_file);
                 $finish;
            end else begin
                 $error("[%0t] MLP Driver: Premature end-of-file in 'input.txt' when expecting second input. Aborting.", $time);
                 $fclose(mcd_input_file);
                 $finish;
            end
        end else begin
            $error("[%0t] MLP Driver: Premature end-of-file in 'input.txt' before reading the second input. Aborting.", $time);
            $fclose(mcd_input_file);
            $finish;
        end
        
        $fclose(mcd_input_file); // Close the input file
        $display("[%0t] MLP Driver: Successfully read inputs from 'input.txt'.", $time);


        $display("[%0t] MLP Driver: Input vector x = [%0d (0x%h), %0d (0x%h)]", $time,
                 tb_inputs[0], tb_inputs[0], tb_inputs[1], tb_inputs[1]);

        // --- Populate Behavioral Model Inputs ---
        //tb_inputs[0] = (-1*(1 << IN_WIDTH/2)); 
        //tb_inputs[1] = (2*(1 << IN_WIDTH/2));
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

    

        $finish;
    end

endmodule