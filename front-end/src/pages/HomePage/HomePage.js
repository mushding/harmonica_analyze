import React, { Component } from 'react';
import '../../assets/css/main.css'
import HomePageText from '../../components/HomePageText/HomePageText'
import Footer from '../../components/Footer/Footer'

export default class HomePage extends Component{
    componentDidMount(){
    }
    render(){
        return (
            <div>
                {/* <ShowWave/> */}
                <div id="wrapper">
                	<header id="header">
						<div className="logo">
							<span className="icon fa-chart-bar"></span>
						</div>
						<div className="content">
							<div className="inner">
								<HomePageText wait={1000}/>
							</div>
						</div>
						<nav>
							<ul>
								<li><a href="/#/start">音色檢測模式</a></li>
								<li><a href="/#/notetest">音高比對模式</a></li>
							</ul>
						</nav>
					</header>	
					<Footer/>
                </div>
            </div>
        );
    }
}