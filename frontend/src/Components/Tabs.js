import React, { useState } from "react";
import TabNavItem from "./TabComponent/TabNavItem";
import TabContent from "./TabComponent/TabContent";
 
const Tabs = () => {
  const [activeTab, setActiveTab] = useState("tab1");
 
  return (
    <div className="Tabs">
        <ul className="nav">
            <TabNavItem title="Tab 1" id="tab1" activeTab={activeTab} setActiveTab={setActiveTab}/>
            <TabNavItem title="Tab 2" id="tab2" activeTab={activeTab} setActiveTab={setActiveTab}/>
            <TabNavItem title="Tab 3" id="tab3" activeTab={activeTab} setActiveTab={setActiveTab}/>
        </ul>
    
        <div className="outlet">
            <TabContent id="tab1" activeTab={activeTab}>
              <p>Tab 1 works!</p>
            </TabContent>
            <TabContent id="tab2" activeTab={activeTab}>
              <p>Tab 2 works!</p>
            </TabContent>
            <TabContent id="tab3" activeTab={activeTab}>
              <p>Tab 3 works!</p>
            </TabContent>
        </div>
    </div>
  );
};
 
export default Tabs;