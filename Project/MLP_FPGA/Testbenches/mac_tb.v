`timescale 1ns / 1ps

module mac_tb;

    parameter A_WIDTH = 8;
    parameter B_WIDTH = 8;
    parameter R_WIDTH = 32;

    reg clk;
    reg start;
    reg valid;
    reg signed [A_WIDTH-1:0] a;
    reg signed [B_WIDTH-1:0] b;

    wire signed [R_WIDTH-1:0] result;

    // Instantiate MAC
    mac #(
        .A_WIDTH(A_WIDTH),
        .B_WIDTH(B_WIDTH),
        .R_WIDTH(R_WIDTH)
    ) uut (
        .clk(clk),
        .start(start),
        .valid(valid),
        .a(a),
        .b(b),
        .result(result)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    // Test vectors
    reg signed [A_WIDTH-1:0] a_vec [0:3];
    reg signed [B_WIDTH-1:0] b_vec [0:3];
    integer i;

    initial begin
        $display("Starting MAC testbench...");
        $dumpfile("mac.vcd");
        $dumpvars(0, mac_tb);

        // Initialize test values
        a_vec[0] = 3;   b_vec[0] = 2;
        a_vec[1] = -1;  b_vec[1] = 5;
        a_vec[2] = 4;   b_vec[2] = -2;
        a_vec[3] = 1;   b_vec[3] = 10;

        a = 0;
        b = 0;
        start = 0;
        valid = 0;

        #20;

        // First input with start = 1
        @(negedge clk);
        a = a_vec[0];
        b = b_vec[0];
        start = 1;

        @(negedge clk);
        start = 0;
        valid = 1;

        // Remaining inputs with valid = 1
        for (i = 1; i < 4; i = i + 1) begin
            @(negedge clk);
            a = a_vec[i];
            b = b_vec[i];
        end

        // End valid
        @(negedge clk);
        valid = 0;

        #20;
        $display("Final accumulated result = %0d", result);
    end

endmodule