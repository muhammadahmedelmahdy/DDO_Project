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

        self.all_waybill_df["grab_time"] = pd.to_datetime(
            self.all_waybill_df["grab_time"], unit="s"
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

    def get_active_orders(self, timestamp):
        """
        Retrieve active orders at the given timestamp.
        """
        timestamp = pd.to_datetime(timestamp, unit="s")
        active_orders = self.all_waybill_df[
            (self.all_waybill_df["dispatch_time"] <= timestamp)
            & (self.all_waybill_df["arrive_time"] > timestamp)
        ]
        return active_orders[
            [
                "order_id",
                "sender_lat",
                "sender_lng",
                "recipient_lat",
                "recipient_lng",
            ]
        ]

    def get_active_couriers(self, timestamp):
        """
        Retrieve active couriers at the given timestamp.
        """
        timestamp = pd.to_datetime(timestamp, unit="s")
        active_couriers = self.courier_wave_df[
            (self.courier_wave_df["wave_start_time"] <= timestamp)
            & (self.courier_wave_df["wave_end_time"] >= timestamp)
        ]
        # Retrieve the most recent location of each courier from all_waybill_df
        courier_locations = (
            self.all_waybill_df[self.all_waybill_df["grab_time"] <= timestamp]
            .sort_values(by="grab_time", ascending=False)
            .drop_duplicates(subset=["courier_id"])[
                ["courier_id", "grab_lat", "grab_lng"]
            ]
        )
        # Merge active couriers with locations
        active_couriers = active_couriers.merge(
            courier_locations, on="courier_id", how="left"
        )
        # Fill missing locations and details with default values
        active_couriers["grab_lat"].fillna(0, inplace=True)
        active_couriers["grab_lng"].fillna(0, inplace=True)
        active_couriers["wave_id"].fillna(-1, inplace=True)
        active_couriers["order_ids"].fillna("[]", inplace=True)
        # Add this at the end of the get_active_couriers method
        active_couriers["unfulfilled_orders"] = 0  # Initialize with 0
        return active_couriers[
            [
                "courier_id",
                "wave_id",
                "order_ids",
                "grab_lat",
                "grab_lng",
                "unfulfilled_orders",
            ]
        ]

        # return active_couriers[
        #     ["courier_id", "wave_id", "order_ids", "grab_lat", "grab_lng"]
        # ]

    def get_unfulfilled_orders(self, courier_order_ids, timestamp):
        """
        Retrieve the number of unfulfilled orders for a courier before the given timestamp.
        """
        if not courier_order_ids:  # Handle case where no orders are assigned
            return 0

        timestamp = pd.to_datetime(timestamp, unit="s")

        # Filter orders assigned to the courier
        unfulfilled_orders = self.all_waybill_df[
            (self.all_waybill_df["order_id"].isin(courier_order_ids))
            & (self.all_waybill_df["arrive_time"] > timestamp)
        ]

        # Return the count of unfulfilled orders
        return len(unfulfilled_orders)

    def construct_state(self, timestamp):
        """
        Construct the RL environment state at the given timestamp.
        """
        # Get active orders
        active_orders = self.get_active_orders(timestamp)

        # Get active couriers and count their unfulfilled orders
        active_couriers = self.get_active_couriers(timestamp)
        courier_details = []
        for _, row in active_couriers.iterrows():
            unfulfilled_count = self.get_unfulfilled_orders(row["order_ids"], timestamp)
            courier_details.append(
                {
                    "courier_id": row["courier_id"],
                    "wave_id": row["wave_id"],
                    "grab_lat": row["grab_lat"],
                    "grab_lng": row["grab_lng"],
                    "unfulfilled_orders": unfulfilled_count,
                }
            )

        # Construct state dictionary
        state = {
            "orders": active_orders.to_dict(orient="records"),
            "couriers": courier_details,
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

    # Define the timestamp
    timestamp = 1666077600  # Example timestamp (Unix time)

    # Construct state and save to JSON
    state = data_module.construct_state(timestamp)
    with open("state.json", "w") as f:
        json.dump(state, f, indent=4)

    print("State saved to 'state.json'")
    print(json.dumps(state, indent=4))
