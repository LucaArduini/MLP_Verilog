module MLP_mac #(
    parameter A_WIDTH = 16,                 // Bit-width of input A
    parameter B_WIDTH = 16,		            // Bit-width of input B
    parameter ACC_WIDTH = 64                // Bit-width of the result/accumulator
)(
    input clk,                              // Clock signal
    input start,                            // Initializes accumulator with a * b
    input valid,                            // Triggers accumulation: acc <= acc + (a * b)
    input signed [A_WIDTH-1:0] a,           // Signed input operand A
    input signed [B_WIDTH-1:0] b,           // Signed input operand B
    output signed [ACC_WIDTH-1:0] result    // MAC operation result
);

    // Internal register to store the accumulated value
    reg signed [ACC_WIDTH-1:0] acc;
    // Wire for the direct product of a and b
    wire signed [A_WIDTH + B_WIDTH - 1:0] product;
    // Wire for the product, sign-extended to match the accumulator's width
    wire signed [ACC_WIDTH-1:0] product_ext;

    // Perform multiplication of inputs a and b
    assign product = a * b;

    // Sign-extend the product to ACC_WIDTH
    // Replicates the sign bit (MSB of 'product') to fill the upper bits
    assign product_ext = {{(ACC_WIDTH - (A_WIDTH + B_WIDTH)){product[A_WIDTH + B_WIDTH - 1]}}, product};

    // Output the current accumulator value
    assign result = (acc >>> A_WIDTH/2);

    // Sequential logic for accumulator updates
    always @(posedge clk) begin
        if (start) begin
            // On 'start' signal, load the (extended) product into the accumulator
            acc <= product_ext;
        end
        else if (valid) begin
            // On 'valid' signal (and not 'start'), add the (extended) product to the accumulator
            acc <= acc + product_ext;
        end
        // Implicit case:
        // else if (start == 0 && valid == 0)
        //     // acc <= acc;           (which is redundant and typically not written)
        //     // 
        //     // Inside an always @(posedge clk) block, if a 'reg' type variable
        //     // is not assigned a new value in any branch of the conditional logic
        //     // for a given clock edge, it retains its previous value.
        //     // This is the fundamental behavior of a memory element (like a flip-flop):
        //     // it "remembers" its state.
        // end

    end

endmodule
