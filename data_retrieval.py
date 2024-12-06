import json
from ast import literal_eval

import pandas as pd


class DataGenerationModule:
    def __init__(
        self,
        courier_wave_path,
        all_waybill_path,
        dispatching_order_path,
        dispatch_waybill_path,
    ):
        # Load datasets
        self.courier_wave_df = pd.read_csv(courier_wave_path)
        self.all_waybill_df = pd.read_csv(all_waybill_path)
        self.dispatching_order_df = pd.read_csv(dispatching_order_path)
        self.dispatch_waybill_df = pd.read_csv(dispatch_waybill_path)

        # Preprocess datasets
        self.preprocess_data()

    def preprocess_data(self):
        # Drop redundant columns
        self.courier_wave_df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        self.all_waybill_df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        self.dispatching_order_df.drop(
            columns=["Unnamed: 0"], inplace=True, errors="ignore"
        )
        self.dispatch_waybill_df.drop(
            columns=["Unnamed: 0"], inplace=True, errors="ignore"
        )

        # Convert string representations of lists to Python lists
        if "order_ids" in self.courier_wave_df.columns:
            self.courier_wave_df["order_ids"] = self.courier_wave_df["order_ids"].apply(
                literal_eval
            )
        if "courier_waybills" in self.dispatching_order_df.columns:
            self.dispatching_order_df["courier_waybills"] = self.dispatching_order_df[
                "courier_waybills"
            ].apply(literal_eval)

        # Convert timestamps to datetime where applicable
        time_columns = [
            "wave_start_time",
            "wave_end_time",
            "dispatch_time",
            "grab_time",
            "fetch_time",
            "arrive_time",
        ]
        for col in time_columns:
            if col in self.all_waybill_df.columns:
                self.all_waybill_df[col] = pd.to_datetime(
                    self.all_waybill_df[col], unit="s"
                )
            if col in self.courier_wave_df.columns:
                self.courier_wave_df[col] = pd.to_datetime(
                    self.courier_wave_df[col], unit="s"
                )

    def get_active_orders(self, start_time, end_time):
        """
        Retrieve active orders within the given time window.
        Active orders are:
        - Dispatched within the time window.
        - Arrival time is after the end of the time window.
        """
        # Convert start and end times to datetime
        start_time = pd.to_datetime(start_time, unit="s")
        end_time = pd.to_datetime(end_time, unit="s")

        # Filter orders based on dispatch and arrival times
        active_orders = self.all_waybill_df[
            (self.all_waybill_df["dispatch_time"] >= start_time)
            & (self.all_waybill_df["dispatch_time"] <= end_time)
            & (self.all_waybill_df["arrive_time"] > end_time)
        ]

        # Return only relevant columns for orders
        return active_orders[
            [
                "order_id",
                "dispatch_time",
                "arrive_time",
                "sender_lat",
                "sender_lng",
                "recipient_lat",
                "recipient_lng",
            ]
        ]

    def get_active_couriers(self, start_time, end_time):
        """
        Retrieve active couriers within the given time window.
        Active couriers are:
        - Those whose wave times overlap with the specified time window.
        """
        # Convert start and end times to datetime
        start_time = pd.to_datetime(start_time, unit="s")
        end_time = pd.to_datetime(end_time, unit="s")

        # Filter couriers based on wave start and end times
        active_couriers = self.courier_wave_df[
            (self.courier_wave_df["wave_start_time"] <= end_time)
            & (self.courier_wave_df["wave_end_time"] >= start_time)
        ]

        # Return only relevant columns for couriers
        return active_couriers[
            ["courier_id", "wave_id", "wave_start_time", "wave_end_time", "order_ids"]
        ]

    def construct_state(self, start_time, end_time):
        """
        Construct the RL environment state for the given time window.
        The state includes active orders, active couriers, and system statistics.
        """
        # Get active orders and couriers
        active_orders = self.get_active_orders(start_time, end_time)
        active_couriers = self.get_active_couriers(start_time, end_time)

        # Construct state dictionary
        state = {
            "orders": active_orders.to_dict(orient="records"),
            "couriers": active_couriers.to_dict(orient="records"),
            "system": {
                "active_orders": len(active_orders),
                "active_couriers": len(active_couriers),
            },
        }
        return state


# Example usage
if __name__ == "__main__":
    # File paths
    courier_wave_path = "courier_wave_info_meituan.csv"
    all_waybill_path = "all_waybill_info_meituan_0322.csv"
    dispatching_order_path = "dispatch_rider_meituan.csv"
    dispatch_waybill_path = "dispatch_waybill_meituan.csv"

    # Initialize the data generation module
    data_module = DataGenerationModule(
        courier_wave_path,
        all_waybill_path,
        dispatching_order_path,
        dispatch_waybill_path,
    )

    # Define the time window
    start_time = 1666077600  # Example start timestamp (Unix time)
    end_time = 1666077900  # Example end timestamp (Unix time)

    # Retrieve active orders and couriers
    active_orders = data_module.get_active_orders(start_time, end_time)
    active_couriers = data_module.get_active_couriers(start_time, end_time)

    print("Active Orders:")
    print(active_orders)
    print("\nActive Couriers:")
    print(active_couriers)

    # Construct state and save to JSON
    state = data_module.construct_state(start_time, end_time)
    with open("state.json", "w") as f:
        json.dump(state, f, indent=4)

    print("\nState saved to 'state.json'")
