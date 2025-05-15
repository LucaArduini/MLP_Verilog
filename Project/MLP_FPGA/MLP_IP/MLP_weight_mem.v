module MLP_weight_mem #(
    parameter ADDR_WIDTH = 6,      // log2(number of perceptrons)
    parameter DATA_WIDTH = 32      // Width of each weight (e.g., fixed-point)
)(
    input 					clk,
	input 					rst,
    input [ADDR_WIDTH-1:0] 	addr,
	input 					wr_en,
	input [DATA_WIDTH-1:0] 	wr_data,
	
    output [DATA_WIDTH-1:0] rd_data
);

    // Memory declaration with M9K block usage
    (* ramstyle = "M9K" *) 
    reg [DATA_WIDTH-1:0] mem [0:(1 << ADDR_WIDTH)-1];

    always @(posedge clk) begin
        if (rst) begin
            // Optional reset behavior (typically no init for BRAM)

        end else if (wr_en) begin
            mem[addr] <= wr_data;
        end
    end
	
	assign rd_data = mem[addr];

endmodule

