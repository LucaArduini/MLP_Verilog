module MLP_layer #(
    parameter N_INPUTS     = 2,
    parameter N_NEURONS    = 4,
    parameter IN_WIDTH     = 16,
    parameter WGT_WIDTH    = 16,
    parameter MAC_WIDTH    = 32,
    parameter OUT_WIDTH    = 16
)(
    input   clk,

    // === Weight memory write interface ===
    input                             wr_en,
    input   [WGT_WIDTH-1:0]           wr_weight,
	input   [$clog2(N_NEURONS)-1:0]   wr_row,
	input   [$clog2(N_INPUTS)-1:0]    wr_col, 
 

    // === Input and control signals ===
    input   signed [IN_WIDTH-1:0]     input_value,
    input   [$clog2(N_INPUTS)-1:0]    input_index,   
    input                             valid,     // triggers MAC operation
    input                             start,     // clears accumulators
    input                             relu_en,   // triggers output registration

    // === Output vector ===
    output reg [N_NEURONS*OUT_WIDTH-1:0] outputs_flat
);

    // === Constants ===
    localparam signed [OUT_WIDTH-1:0] MAX_VAL = {1'b0, {(OUT_WIDTH-1){1'b1}}};

    // === Weight memory outputs ===
    wire signed [WGT_WIDTH-1:0] weight_out [N_NEURONS-1:0];

    // === MAC outputs ===
    wire signed [MAC_WIDTH-1:0] mac_outputs [N_NEURONS-1:0];
	
    // === Instantiate weight memories ===
	genvar j;
	generate
		for (j = 0; j < N_NEURONS; j = j + 1) begin : gen_weight_mem
			MLP_weight_mem #(
				.ADDR_WIDTH($clog2(N_INPUTS)),
				.DATA_WIDTH(WGT_WIDTH)
			) weight_mem_j (
				.clk(clk),
				.rst(1'b0),
				.wr_en(wr_en && (wr_row == j)),
				.addr(wr_en ? wr_col : input_index),
				.wr_data(wr_weight),
				.rd_data(weight_out[j])
			);
		end
	endgenerate

	// === Connect weights to MACs ===
	generate
		for (j = 0; j < N_NEURONS; j = j + 1) begin : gen_mac
			MLP_mac #(
				.A_WIDTH(IN_WIDTH),
				.B_WIDTH(WGT_WIDTH),
				.ACC_WIDTH(MAC_WIDTH)
			) mac_j (
				.clk(clk),
				.a(input_value),     // same input broadcasted
				.b(weight_out[j]),   // weight for neuron j
				.valid(valid),
				.start(start),
				.result(mac_outputs[j])
			);
		end
	endgenerate

    // === ReLU + Clipping stage ===  
	always @(posedge clk) begin
	    integer n;
		if (relu_en) begin
			for (n = 0; n < N_NEURONS; n = n + 1) begin
				if (mac_outputs[n] < 0)
					outputs_flat[n*OUT_WIDTH +: OUT_WIDTH] <= 0;
				else if (mac_outputs[n] > MAX_VAL)
					outputs_flat[n*OUT_WIDTH +: OUT_WIDTH] <= MAX_VAL;
				else
					outputs_flat[n*OUT_WIDTH +: OUT_WIDTH] <= mac_outputs[n][OUT_WIDTH-1:0];
			end
		end
	end


endmodule