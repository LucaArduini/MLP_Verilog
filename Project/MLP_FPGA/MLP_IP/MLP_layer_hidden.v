module MLP_layer_hidden #(
    parameter N_INPUTS     = 2+1,                   // Number of inputs to the layer (and thus, weights per neuron)
    parameter N_NEURONS    = 4,                     // Number of neurons in this layer
    parameter IN_WIDTH     = 16,                    // Bit-width of each input value
    parameter WGT_WIDTH    = 16,                    // Bit-width of each weight value
    parameter MAC_WIDTH    = 64,                    // Bit-width of the accumulator in the MAC units
    parameter OUT_WIDTH    = 16                     // Bit-width of each neuron's output (after ReLU/clipping)
)(
    input   clk,                                    // Clock signal

    // === Weight memory write interface ===
    // Allows external loading of weights into the neuron's weight memories.
    input                             wr_en,        // Global write enable for all weight memories
    input   [WGT_WIDTH-1:0]           wr_weight,    // Weight data to be written
	input   [$clog2(N_NEURONS)-1:0]   wr_row,       // Selects which neuron's weight memory to write to (0 to N_NEURONS-1)
	input   [$clog2(N_INPUTS)-1:0]    wr_col,       // Selects the specific weight address within the chosen neuron's memory (0 to N_INPUTS-1)
 

    // === Input and control signals ===
    input   signed [IN_WIDTH-1:0]     input_value,   // Current input value being processed
    input   [$clog2(N_INPUTS)-1:0]    input_index,   // Index of the current input_value (0 to N_INPUTS-1), used as read address for weight memories
    input                             start,         // Signal to initialize/reset all MAC accumulators (acc = input * weight, or 0 if first input is 0)
    input                             valid,         // Signal to trigger the accumulation step in all MAC units (acc += input * weight)
    input                             relu_en,       // Enable signal to register the MAC outputs through ReLU/clipping stage into `outputs_flat`

    // === Output vector ===
    // All neuron outputs are concatenated into a single flat vector.
    output reg [N_NEURONS*OUT_WIDTH-1:0] outputs_flat
);

    // === Constants ===
    // Maximum positive value representable by OUT_WIDTH bits (signed, but used for positive clipping)
    localparam signed [OUT_WIDTH-1:0] MAX_VAL = {1'b0, {(OUT_WIDTH-1){1'b1}}}; // e.g., 0111...1

    // === Weight memory outputs ===
    // Array of wires to hold the weight read from each neuron's dedicated weight memory.
    // `weight_out[j]` is the weight for the current `input_index` for neuron `j`.
    wire signed [WGT_WIDTH-1:0] weight_out [N_NEURONS-1:0];

    // === MAC outputs ===
    // Array of wires to hold the accumulated result from each neuron's MAC unit.
    wire signed [MAC_WIDTH-1:0] mac_outputs [N_NEURONS-1:0];
	
    // === Instantiate weight memories ===
    // Generates N_NEURONS instances of MLP_weight_mem, one for each neuron.
	genvar j; // Loop variable for generation
	generate
		for (j = 0; j < N_NEURONS; j = j + 1) begin : gen_weight_mem
			MLP_weight_mem #(
				.ADDR_WIDTH($clog2(N_INPUTS)), // Each memory stores N_INPUTS weights
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
    // Generates N_NEURONS instances of MLP_mac, one for each neuron.
	generate
		for (j = 0; j < N_NEURONS; j = j + 1) begin : gen_mac
			MLP_mac #(
				.A_WIDTH(IN_WIDTH),
				.B_WIDTH(WGT_WIDTH),
				.ACC_WIDTH(MAC_WIDTH)
			) mac_j (
				.clk(clk),
				.a(input_value),        // The current input_value is broadcasted to all MAC units.
				.b(weight_out[j]),      // The corresponding weight for neuron 'j' (from its weight memory).
				.valid(valid),          // Pass through valid signal
				.start(start),          // Pass through start signal
				.result(mac_outputs[j]) // Accumulated result for neuron j
			);
		end
	endgenerate

    // === ReLU + Clipping stage ===
    // This block registers the MAC outputs after applying ReLU and clipping, when 'relu_en' is asserted.
	always @(posedge clk) begin : relu_clip_logic
	    integer n; // Loop variable for neurons
		if (relu_en) begin
			for (n = 0; n < N_NEURONS; n = n + 1) begin
                // NOTE: This `for` loop does not execute sequentially over multiple clock cycles.
                // During synthesis, it is "unrolled" to create N_NEURONS parallel copies of
                // this logic. All comparisons happen concurrently, and the entire update
                // completes within a single clock cycle when `relu_en` is asserted.

				if (mac_outputs[n] < 0) begin                       // ReLU: if negative, output 0
					outputs_flat[n*OUT_WIDTH +: OUT_WIDTH] <= 0;
				end
                else if (mac_outputs[n] > MAX_VAL) begin            // Clipping: if greater than MAX_VAL, output MAX_VAL
                                                                    // Note: This comparison implicitly extends MAX_VAL to MAC_WIDTH for comparison.
					outputs_flat[n*OUT_WIDTH +: OUT_WIDTH] <= MAX_VAL;
				end
                else begin                                          // Otherwise, output the (truncated) MAC result
					outputs_flat[n*OUT_WIDTH +: OUT_WIDTH] <= mac_outputs[n][OUT_WIDTH-1:0];
				end
			end
		end
        // If relu_en is not asserted, outputs_flat retains its previous value.
	end

endmodule
