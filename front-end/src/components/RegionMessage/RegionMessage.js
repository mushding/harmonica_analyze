import React from 'react'
import './styles.css'
export default class RegionMessage extends React.Component{
    componentDidMount(){
    }
    render(){
        const region = this.props.region
        return(
            <div className="messageContainer">
                <div className="textContainer">
                    { region.attributes === "1" ? (
                        <div>
                            <h3>這裡的音色吹得不太飽滿喔</h3>
                            <h3>試著把嘴吧張大一點，把琴格都蓋滿</h3>
                        </div>
                    ) : region.attributes === "2" ? (
                        <div>
                            <h3>這裡有吹到兩個以上的音喔</h3>
                            <h3>跳位練習可以多加強一些！</h3>
                        </div>
                    ) : region.attributes === "4" ? (
                        <div>
                            <h3>這裡換氣的聲音太大聲囉</h3>
                            <h3>慢慢的吸氣，換氣的時候嘴吧不用離口琴太遠喔</h3>
                        </div>
                    ) : (
                        <h3>attributes: {region.attributes}</h3>
                    )}
                </div>
                <div className="textContainer">
                    <h3>start at {region.start} sec</h3>
                </div>
                <div className="textContainer">
                    <h3>end at {region.end} sec</h3>
                </div>
            </div>
        );
    }
}