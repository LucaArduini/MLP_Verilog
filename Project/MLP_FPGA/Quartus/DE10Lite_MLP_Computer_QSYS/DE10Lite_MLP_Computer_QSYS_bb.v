
module DE10Lite_MLP_Computer_QSYS (
	clk_clk,
	clk_sdram_clk,
	hex3_hex0_external_connection_export,
	hex5_hex4_external_connection_export,
	key_external_connection_export,
	ledr_external_connection_export,
	reset_reset_n,
	sdram_wire_addr,
	sdram_wire_ba,
	sdram_wire_cas_n,
	sdram_wire_cke,
	sdram_wire_cs_n,
	sdram_wire_dq,
	sdram_wire_dqm,
	sdram_wire_ras_n,
	sdram_wire_we_n,
	sliders_external_connection_export);	

	input		clk_clk;
	output		clk_sdram_clk;
	output	[31:0]	hex3_hex0_external_connection_export;
	output	[15:0]	hex5_hex4_external_connection_export;
	input	[1:0]	key_external_connection_export;
	output	[9:0]	ledr_external_connection_export;
	input		reset_reset_n;
	output	[12:0]	sdram_wire_addr;
	output	[1:0]	sdram_wire_ba;
	output		sdram_wire_cas_n;
	output		sdram_wire_cke;
	output		sdram_wire_cs_n;
	inout	[15:0]	sdram_wire_dq;
	output	[1:0]	sdram_wire_dqm;
	output		sdram_wire_ras_n;
	output		sdram_wire_we_n;
	input	[9:0]	sliders_external_connection_export;
endmodule
