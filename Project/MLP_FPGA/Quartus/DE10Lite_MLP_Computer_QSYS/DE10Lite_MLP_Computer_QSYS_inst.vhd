	component DE10Lite_MLP_Computer_QSYS is
		port (
			clk_clk                              : in    std_logic                     := 'X';             -- clk
			clk_sdram_clk                        : out   std_logic;                                        -- clk
			hex3_hex0_external_connection_export : out   std_logic_vector(31 downto 0);                    -- export
			hex5_hex4_external_connection_export : out   std_logic_vector(15 downto 0);                    -- export
			key_external_connection_export       : in    std_logic_vector(1 downto 0)  := (others => 'X'); -- export
			ledr_external_connection_export      : out   std_logic_vector(9 downto 0);                     -- export
			reset_reset_n                        : in    std_logic                     := 'X';             -- reset_n
			sdram_wire_addr                      : out   std_logic_vector(12 downto 0);                    -- addr
			sdram_wire_ba                        : out   std_logic_vector(1 downto 0);                     -- ba
			sdram_wire_cas_n                     : out   std_logic;                                        -- cas_n
			sdram_wire_cke                       : out   std_logic;                                        -- cke
			sdram_wire_cs_n                      : out   std_logic;                                        -- cs_n
			sdram_wire_dq                        : inout std_logic_vector(15 downto 0) := (others => 'X'); -- dq
			sdram_wire_dqm                       : out   std_logic_vector(1 downto 0);                     -- dqm
			sdram_wire_ras_n                     : out   std_logic;                                        -- ras_n
			sdram_wire_we_n                      : out   std_logic;                                        -- we_n
			sliders_external_connection_export   : in    std_logic_vector(9 downto 0)  := (others => 'X')  -- export
		);
	end component DE10Lite_MLP_Computer_QSYS;

	u0 : component DE10Lite_MLP_Computer_QSYS
		port map (
			clk_clk                              => CONNECTED_TO_clk_clk,                              --                           clk.clk
			clk_sdram_clk                        => CONNECTED_TO_clk_sdram_clk,                        --                     clk_sdram.clk
			hex3_hex0_external_connection_export => CONNECTED_TO_hex3_hex0_external_connection_export, -- hex3_hex0_external_connection.export
			hex5_hex4_external_connection_export => CONNECTED_TO_hex5_hex4_external_connection_export, -- hex5_hex4_external_connection.export
			key_external_connection_export       => CONNECTED_TO_key_external_connection_export,       --       key_external_connection.export
			ledr_external_connection_export      => CONNECTED_TO_ledr_external_connection_export,      --      ledr_external_connection.export
			reset_reset_n                        => CONNECTED_TO_reset_reset_n,                        --                         reset.reset_n
			sdram_wire_addr                      => CONNECTED_TO_sdram_wire_addr,                      --                    sdram_wire.addr
			sdram_wire_ba                        => CONNECTED_TO_sdram_wire_ba,                        --                              .ba
			sdram_wire_cas_n                     => CONNECTED_TO_sdram_wire_cas_n,                     --                              .cas_n
			sdram_wire_cke                       => CONNECTED_TO_sdram_wire_cke,                       --                              .cke
			sdram_wire_cs_n                      => CONNECTED_TO_sdram_wire_cs_n,                      --                              .cs_n
			sdram_wire_dq                        => CONNECTED_TO_sdram_wire_dq,                        --                              .dq
			sdram_wire_dqm                       => CONNECTED_TO_sdram_wire_dqm,                       --                              .dqm
			sdram_wire_ras_n                     => CONNECTED_TO_sdram_wire_ras_n,                     --                              .ras_n
			sdram_wire_we_n                      => CONNECTED_TO_sdram_wire_we_n,                      --                              .we_n
			sliders_external_connection_export   => CONNECTED_TO_sliders_external_connection_export    --   sliders_external_connection.export
		);

