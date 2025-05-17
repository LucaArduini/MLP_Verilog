
`timescale 1ns / 100ps

module mlp_tb;

    // Parameters
    localparam N_INPUTS     = 2;
    localparam N_HIDDEN     = 4;
    localparam N_OUTPUT     = 1;
    localparam IN_WIDTH     = 16;
    localparam WGT_WIDTH    = 16;
    localparam MAC_WIDTH    = 64;
    localparam OUT_WIDTH    = 16;

    localparam CLK_PERIOD = 10;

    // Testbench signals
    reg clk, rst;
    reg write_en;
    reg [1:0] addr;
    reg [31:0] writedata;
    wire [31:0] readdata;
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

    // Initial stimulus
    initial begin
        $dumpfile("mlp.vcd");
        $dumpvars(0, mlp_tb);

        clk = 0;
        rst = 1;
        write_en = 0;
        addr = 0;
        writedata = 0;

        repeat(2) @(posedge clk);
        rst = 0;

        // === Load input vector: x = [7, -3] ===
        // Bias is implicit and handled in hardware
        write_reg(2'd1, 32'sd7);
        write_reg(2'd1, -32'sd3);

        // === Load hidden layer weights (4 neurons × 3 inputs) ===
        // Format: [bias, x0, x1]
        // Neuron 0
        write_reg(2'd2, 32'sd1);  // bias
        write_reg(2'd2, 32'sd2);  // x0
        write_reg(2'd2, 32'sd3);  // x1
        // Neuron 1
        write_reg(2'd2, 32'sd0);
        write_reg(2'd2, -32'sd1);
        write_reg(2'd2, 32'sd2);
        // Neuron 2
        write_reg(2'd2, -32'sd2);
        write_reg(2'd2, 32'sd4);
        write_reg(2'd2, 32'sd1);
        // Neuron 3
        write_reg(2'd2, 32'sd1);
        write_reg(2'd2, 32'sd1);
        write_reg(2'd2, 32'sd1);

        
        // === Load output layer weights (1 neuron × 5 inputs) ===
        // Format: [bias, h0, h1, h2, h3]

        write_reg(2'd0, 32'sb1000); // change layer

        write_reg(2'd2, 32'sd1);  // bias
        write_reg(2'd2, 32'sd1);
        write_reg(2'd2, 32'sd1);
        write_reg(2'd2, 32'sd1);
        write_reg(2'd2, 32'sd1);

        // === Start MLP computation ===
        write_reg(2'd0, 32'b00000001);  // Set RUN bit

        // Wait for DONE (bit 1 in CTRL)
        wait(dut.ctrl[1] == 1);

        // === Read output ===
        write_reg(2'd3, 32'd0);  // Output select if needed
        @(posedge clk);
        $display("MLP output: %0d", readdata);

        $finish;
    end

endmodule
