module MLP_mac #(
    parameter A_WIDTH = 8,   // Bit-width of input A
    parameter B_WIDTH = 8,   // Bit-width of input B
    parameter ACC_WIDTH = 32   // Bit-width of the result/accumulator
)(
    input clk,
    input start,      // Initializes acc = a * b
    input valid,      // Triggers accumulation: acc += a * b
    input signed [A_WIDTH-1:0] a,
    input signed [B_WIDTH-1:0] b,
    output signed [ACC_WIDTH-1:0] result
);

    reg signed [ACC_WIDTH-1:0] acc;
    wire signed [A_WIDTH + B_WIDTH - 1:0] product;
    wire signed [ACC_WIDTH-1:0] product_ext;

    assign product = a * b;

    // Explicit sign extension to ACC_WIDTH
    assign product_ext = {{(ACC_WIDTH - (A_WIDTH + B_WIDTH)){product[A_WIDTH + B_WIDTH - 1]}}, product};

    assign result = acc;

    always @(posedge clk) begin
        if (start) begin
            acc <= product_ext;
        end else if (valid) begin
            acc <= acc + product_ext;
        end
        // Implicit case:
        // else if (start == 0 && valid == 0)
        //     // acc <= acc; (which is redundant and typically not written)
        //     // Inside an always @(posedge clk) block, if a 'reg' type variable
        //     // is not assigned a new value in any branch of the conditional logic
        //     // for a given clock edge, it retains its previous value.
        //     // This is the fundamental behavior of a memory element (like a flip-flop):
        //     // it "remembers" its state.
        // end
    end

endmodule