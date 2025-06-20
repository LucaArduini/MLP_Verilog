
module mlp #(
    parameter N_INPUTS     = 2,
    parameter N_HIDDEN     = 4,
    parameter N_OUTPUT     = 1,
    parameter IN_WIDTH     = 16,
    parameter WGT_WIDTH    = 16,
    parameter MAC_WIDTH    = 64,
    parameter OUT_WIDTH    = 16
)(
    input  wire clk,
    input  wire rst,

    // Avalon-MM style interface (simplified)
    input  wire        write_en,
    input  wire [1:0]  addr,
    input  wire [31:0] writedata,
    output wire [31:0] readdata,
    output wire        irq
);
// The first input of each layer is hardcoded to 1.
// This allows the first column of the weight matrix to represent the bias terms.
	localparam N_INPUTS_HIDDEN = N_INPUTS + 1;
	localparam N_INPUTS_OUTPUT = N_HIDDEN + 1;
	

	// === Address map
	localparam ADDR_CTRL   = 2'd0;
	localparam ADDR_INPUT  = 2'd1;
	localparam ADDR_WEIGHT = 2'd2;
	localparam ADDR_OUTPUT = 2'd3;
	
	localparam CTRL_NUM_BITS = 4;
	localparam CTRL_RUN_BIT = 0;
	localparam CTRL_DONE_BIT = 1;
	localparam CTRL_INTERRUPT_BIT = 2;
	localparam CTRL_SET_LAYER_BIT = 3;
	
    // === Internal Registers ===
    reg [CTRL_NUM_BITS-1:0] ctrl;
	reg [$clog2(N_OUTPUT)-1:0] out_sel;
	
    reg signed [IN_WIDTH-1:0] input_regs [0:N_INPUTS_HIDDEN-1];
	
    reg [$clog2(N_INPUTS_HIDDEN):0] input_index;
	reg [$clog2(N_INPUTS_OUTPUT):0] hidden_index;

	// to access weigth matrix of layer 0 (hidden)
    reg [$clog2(N_HIDDEN)-1:0] hidden_weight_row;
    reg [$clog2(N_INPUTS_HIDDEN)-1:0] hidden_weight_col;
	
	// to access weigth matrix of layer 1 (output)
    reg [$clog2(N_OUTPUT)-1:0] output_weight_row;
    reg [$clog2(N_INPUTS_OUTPUT)-1:0] output_weight_col;
	

	// ===
    wire signed [OUT_WIDTH-1:0] hidden_to_output_input;
    wire [N_OUTPUT*OUT_WIDTH-1:0] output_layer_outputs_flat;
    wire [N_HIDDEN*OUT_WIDTH-1:0] hidden_layer_outputs_flat;
	wire [N_INPUTS_OUTPUT*OUT_WIDTH-1:0] output_layer_inputs_flat;

    reg  start_layer_0;
	reg  start_layer_1;
	reg  valid_layer_0;
	reg  valid_layer_1;
    reg  relu_hidden;
    reg  relu_output;
 
	wire restart;
	
		// === FSM State Machine ===
	reg [3:0] state;

	localparam START           = 4'd0,  // Initial state after reset
			   IDLE            = 4'd1,  // Wait for software to start computation
			   PRE_RUN_LAYER0  = 4'd2,  // Reset input index for layer 0
			   RUN_LAYER0      = 4'd3,  // Feed inputs to hidden layer
			   RUN_ReLU0       = 4'd4,  // One-cycle pulse to register hidden layer outputs
			   PRE_RUN_LAYER1  = 4'd5,  // Reset input index for output layer
			   RUN_LAYER1      = 4'd6,  // Feed hidden outputs to output layer
			   RUN_ReLU1       = 4'd7,  // One-cycle pulse to register output layer outputs
			   DONE            = 4'd8;  // Wait for software to acknowledge completion

	always @(posedge clk) begin
		if (rst || restart) begin
			state <= START;
		end else begin
			case (state)
				START: begin
					// Transition immediately to idle
					state <= IDLE;
				end

				IDLE: begin
					// Wait for software to set the run bit
					if (ctrl[CTRL_RUN_BIT])
						state <= PRE_RUN_LAYER0;
				end

				PRE_RUN_LAYER0: begin
					// Reset index before running layer 0
					state <= RUN_LAYER0;
				end

				RUN_LAYER0: begin
					// Process inputs to hidden layer
					if (input_index == N_INPUTS_HIDDEN - 1)
						state <= RUN_ReLU0;
				end

				RUN_ReLU0: begin
					// One-cycle enable to store hidden layer outputs
					state <= PRE_RUN_LAYER1;
				end

				PRE_RUN_LAYER1: begin
					// Reset index before running layer 1
					state <= RUN_LAYER1;
				end

				RUN_LAYER1: begin
					// Process hidden outputs to final output layer
					if (hidden_index == N_INPUTS_OUTPUT - 1)
						state <= RUN_ReLU1;
				end

				RUN_ReLU1: begin
					// One-cycle enable to store final outputs
					state <= DONE;
				end

				DONE: begin
					// Wait for software to clear done flag
					if (restart)
						state <= START;
				end

				default: begin
					state <= START;
				end
			endcase
		end
	end
	
	// Writing address decoding
	// === Address Decoder ===
	reg wr_ctrl, wr_input, wr_weight;

	always @* begin
		// Default: no write
		wr_ctrl   = 1'b0;
		wr_input  = 1'b0;
		wr_weight = 1'b0;

		case (addr)
			ADDR_CTRL:   wr_ctrl   = write_en;
			ADDR_INPUT:  wr_input  = write_en && (state == IDLE);
			ADDR_WEIGHT: wr_weight = write_en  && (state == IDLE);
			default: ; // nothing
		endcase
	end
	
	// === Control reg
	always @(posedge clk) begin
		if (rst) begin
			ctrl <= 0;
		end else begin
			if (wr_ctrl) begin
				ctrl <= writedata[CTRL_NUM_BITS-1:0];
				out_sel <= writedata[31:16];
			end else begin
				case (state)
					START: begin
						ctrl[CTRL_RUN_BIT] <= 1'b0;
						ctrl[CTRL_DONE_BIT] <= 1'b0;
					end
					DONE: begin
						ctrl[CTRL_RUN_BIT] <= 1'b0;
						ctrl[CTRL_DONE_BIT] <= 1'b1;
					end
				endcase
			end
		end
	end
	
	// === input and input_index regs ===
	always @(posedge clk) begin
		if (rst) begin
			input_regs[0] <= 16'b0000000100000000; // 1 in Q7.8 - Bias term for layer 0
			input_index <= 1;
		end else begin
			case (state)
				START: begin
					input_index <= 1;
				end
				IDLE: begin
					if (wr_input) begin
						input_regs[input_index] <= writedata[IN_WIDTH-1:0];
						input_index <= input_index + 1;
					end
				end
				PRE_RUN_LAYER0: begin
					input_index <=0;
				end
				RUN_LAYER0: begin
					input_index <= input_index + 1;
				end	
			endcase
		end
	end
	
	// === hidden_index regs ===
	always @(posedge clk) begin
		if (rst) begin
			hidden_index <= 0;
		end else begin
			case (state)
				PRE_RUN_LAYER1: begin
					hidden_index <=0;
				end
				RUN_LAYER1: begin
					hidden_index <= hidden_index + 1;
				end	
			endcase
		end
	end

	// === Writing weight matrices ===
	always @(posedge clk) begin
		if (rst) begin
			hidden_weight_row <= 0;
			hidden_weight_col <= 0;
			output_weight_row <= 0;
			output_weight_col <= 0;
		end else begin
			case (state)
				START: begin
					hidden_weight_row <= 0;
					hidden_weight_col <= 0;
					output_weight_row <= 0;
					output_weight_col <= 0;
				end
				IDLE: begin
					if (wr_weight) begin
						if (~ctrl[CTRL_SET_LAYER_BIT]) begin
							// layer 0
							if (hidden_weight_col == N_INPUTS_HIDDEN - 1) begin
								hidden_weight_col <= 0;
								if (hidden_weight_row == N_HIDDEN - 1) begin
									hidden_weight_row <= 0;
								end else begin 
									hidden_weight_row <= hidden_weight_row + 1;
								end
							end else begin
								hidden_weight_col <= hidden_weight_col + 1;
							end
						end else begin
							// layer 1
							if (output_weight_col == N_INPUTS_OUTPUT - 1) begin
								output_weight_col <= 0;
								if (output_weight_row == N_OUTPUT - 1) begin
									output_weight_row <= 0;
								end else begin 
									output_weight_row <= output_weight_row + 1;
								end
							end else begin
								output_weight_col <= output_weight_col + 1;
							end
						end
					end
				end
			endcase
		end
	end
	
	// === Control signals ====
	
	assign restart = wr_ctrl && ctrl[CTRL_DONE_BIT];
	
	always @* begin
		// Default
		start_layer_0 = 1'b0;
		start_layer_1 = 1'b0;
		valid_layer_0 = 1'b0;
		valid_layer_1 = 1'b0;
		relu_hidden = 1'b0;
		relu_output = 1'b0;
		
		case (state)
            RUN_LAYER0: begin
				if (input_index == 0) begin
					start_layer_0 = 1'b1;
				end
                valid_layer_0 = 1'b1;
            end

            RUN_ReLU0: begin
                relu_hidden = 1'b1;
            end

			RUN_LAYER1: begin
                // Process hidden outputs to final output layer
                if (hidden_index == 0) begin
                    start_layer_1 = 1'b1;
				end
				valid_layer_1 = 1'b1;
            end

            RUN_ReLU1: begin
                relu_output = 1'b1;
            end
		endcase
	end
	
    // === IRQ ===
    assign irq = ctrl[2] && ctrl[1]; // irq_en && done

    // === MLP Layers ===

    // Layer 0: input -> hidden
    MLP_layer_hidden #(
        .N_INPUTS(N_INPUTS_HIDDEN),
        .N_NEURONS(N_HIDDEN),
        .IN_WIDTH(IN_WIDTH),
        .WGT_WIDTH(WGT_WIDTH),
        .MAC_WIDTH(MAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) layer0 (
        .clk(clk),
        .wr_en(wr_weight && ~ctrl[CTRL_SET_LAYER_BIT]),
        .wr_row(hidden_weight_row),
        .wr_col(hidden_weight_col),
        .wr_weight(writedata[WGT_WIDTH-1:0]),
        .input_value(input_regs[input_index]),
        .input_index(input_index),
        .valid(valid_layer_0),
        .start(start_layer_0),
        .relu_en(relu_hidden),
        .outputs_flat(hidden_layer_outputs_flat)
    );
	
	assign output_layer_inputs_flat = {hidden_layer_outputs_flat,16'b0000000100000000}; // Add bias term (1) to the inputs of the output layer

    // Layer 1: hidden -> output
    MLP_layer_output #(
        .N_INPUTS(N_INPUTS_OUTPUT),
        .N_NEURONS(N_OUTPUT),
        .IN_WIDTH(OUT_WIDTH),
        .WGT_WIDTH(WGT_WIDTH),
        .MAC_WIDTH(MAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) layer1 (
        .clk(clk),
        .wr_en(wr_weight && ctrl[CTRL_SET_LAYER_BIT]),
        .wr_row(output_weight_row),
        .wr_col(output_weight_col),
        .wr_weight(writedata[WGT_WIDTH-1:0]),
        .input_value(output_layer_inputs_flat[hidden_index*OUT_WIDTH +: OUT_WIDTH]),
        .input_index(hidden_index),
        .valid(valid_layer_1),
        .start(start_layer_1),
        .output_en(relu_output),
        .outputs_flat(output_layer_outputs_flat)
    );


// === Registered Read interface ===
reg [31:0] readdata_reg;
assign readdata = readdata_reg;

always @(posedge clk) begin
    if (rst) begin
        readdata_reg <= 32'd0;
    end else begin
        case (addr)
            ADDR_CTRL:   readdata_reg <= {out_sel, {(16-CTRL_NUM_BITS){1'b0}}, ctrl};
            ADDR_OUTPUT: readdata_reg <= output_layer_outputs_flat[out_sel * OUT_WIDTH +: OUT_WIDTH];
            default:     readdata_reg <= 32'd0;
        endcase
    end
end


endmodule
