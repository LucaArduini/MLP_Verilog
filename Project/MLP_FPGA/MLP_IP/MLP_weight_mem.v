module MLP_weight_mem #(
    parameter ADDR_WIDTH = 6,           // log2(number of perceptrons)
    parameter DATA_WIDTH = 16           // Width of each weight (e.g., fixed-point)
)(
    input 					clk,        // Clocòk signal
	input 					rst,        // Reset signal
    input [ADDR_WIDTH-1:0] 	addr,       // Address input for read/write operations
	input 					wr_en,      // Write enable signal
	input [DATA_WIDTH-1:0] 	wr_data,    // Data to be written into the memory

    output [DATA_WIDTH-1:0] rd_data     // Data read from the memory at 'addr'
);

    // Memory declaration with M9K block usage
    // Internal memory (array of registers) with 2^ADDR_WIDTH locations, each location being DATA_WIDTH bits wide
    (* ramstyle = "M9K" *) 
    reg [DATA_WIDTH-1:0] mem [0:(1 << ADDR_WIDTH)-1];

    always @(posedge clk) begin
        if (rst) begin
            //
            // Optional reset behavior (typically no init for BRAM)
            //
        end
        else if (wr_en) begin
            // If write enable is asserted, write 'wr_data' to the location specified by 'addr'
            mem[addr] <= wr_data;
        end
    end
	
	assign rd_data = mem[addr];

endmodule
